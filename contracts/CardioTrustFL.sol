// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title CardioTrustFL v4 -- verifiable, aggregator-free federated learning rounds
/// @notice One round = COMMIT -> REVEAL -> VOTE -> (finalised | failed) -> settle.
///  * Commitments are bound to (sender, round, chain, contract) so they cannot be
///    copied or replayed by another participant (front-running / free-riding).
///  * Reveals are only accepted before a block deadline. Committing without
///    revealing forfeits the deposit and costs reputation (selective-abort penalty).
///  * Every hospital recomputes the deterministic aggregate off-chain over the
///    on-chain ordered set of revealed updates and votes on
///    decision = keccak256(globalHash, keptBitmap, thresholdMilli).
///    Voting is one hospital, one vote (reputation does NOT weight votes).
///    The round finalises when one decision collects q = N - floor((N-1)/3)
///    votes, where N is the number of active hospitals snapshotted when the
///    round starts. If at most f = floor((N-1)/3) of them are Byzantine, a
///    decision needs at least one honest vote (safety) and the N - f honest
///    hospitals alone reach q (liveness). Reputation only gates eligibility.
///  * Reputation is driven by the consensus outcome: revealed-and-kept +5,
///    revealed-but-excluded -5, committed-but-not-revealed -15 and slashed.
contract CardioTrustFL {
    // ------------------------------------------------------------------ types
    enum Phase { Idle, Commit, Reveal, Vote, Finalized, Failed }

    struct Hospital {
        bool registered;
        bool active;
        uint32 index;          // position in hospitalList
        uint64 reputation;     // 0..100
        uint32 rounds;         // rounds with a valid reveal
        uint32 missedReveals;
        uint32 excluded;       // times excluded by consensus
    }

    struct Round {
        uint64 commitEnd;
        uint64 revealEnd;
        uint64 voteEnd;
        Phase phase;
        uint32 nCommit;
        uint32 nReveal;
        uint256 quorum;        // N - floor((N-1)/3), N = active hospitals at start
        bytes32 prevGlobal;
        bytes32 decision;      // winning decision
        bytes32 globalHash;    // agreed global model hash
        uint256 keptBitmap;    // bit i = i-th revealer's update was aggregated
        uint32 thresholdMilli; // federated decision threshold x 1000
    }

    struct RevealData {
        bytes32 updateHash;    // SHA3-256 of the canonical serialised update
        bytes32 uri;           // content address of the update in shared storage
        uint32 epsMilli;       // cumulative (eps, delta)-DP spent by this site, x 1000
        uint32 weight;         // declared aggregation weight
        uint32 order;          // position in the round's reveal order
        bool revealed;
    }

    // ------------------------------------------------------------------ state
    address public governor;
    uint256 public deposit;
    uint64 public commitPeriod;
    uint64 public revealPeriod;
    uint64 public votePeriod;
    uint64 public constant REP_MAX = 100;
    uint64 public minReputation = 20;
    uint256 public activeCount;

    mapping(address => Hospital) public hospitals;
    address[] public hospitalList;

    uint256 public currentRound;
    bytes32 public latestGlobal;
    mapping(uint256 => Round) public rounds;
    mapping(uint256 => mapping(address => bytes32)) public commitments;
    mapping(uint256 => mapping(address => RevealData)) public reveals;
    mapping(uint256 => address[]) public revealOrder;
    mapping(uint256 => mapping(address => bytes32)) public votes;
    mapping(uint256 => mapping(bytes32 => uint256)) public tally;
    mapping(uint256 => mapping(address => bool)) public settled;
    uint256 public slashedPool;

    // ------------------------------------------------------------------ events
    event HospitalRegistered(address indexed h, string name);
    event RoundStarted(uint256 indexed round, uint64 commitEnd, uint64 revealEnd, uint64 voteEnd);
    event Committed(uint256 indexed round, address indexed h, bytes32 c);
    /// The salt is emitted because it also seeds the key that decrypts the blob.
    event Revealed(uint256 indexed round, address indexed h, bytes32 updateHash, bytes32 uri, uint32 epsMilli, uint32 weight, uint32 order, bytes32 salt);
    event Voted(uint256 indexed round, address indexed h, bytes32 decision, uint256 weight);
    event Finalized(uint256 indexed round, bytes32 globalHash, uint256 keptBitmap, uint32 thresholdMilli);
    event RoundFailed(uint256 indexed round);
    event Settled(uint256 indexed round, address indexed h, int256 repDelta, bool slashed);

    modifier onlyGovernor() { require(msg.sender == governor, "governor"); _; }
    modifier onlyActive() { require(hospitals[msg.sender].active, "inactive"); _; }

    constructor(uint256 _deposit, uint64 _commit, uint64 _reveal, uint64 _vote) {
        governor = msg.sender;
        deposit = _deposit;
        commitPeriod = _commit;
        revealPeriod = _reveal;
        votePeriod = _vote;
    }

    // ------------------------------------------------------------ governance
    function registerHospital(address h, string calldata name) external onlyGovernor {
        require(!hospitals[h].registered, "exists");
        hospitals[h] = Hospital(true, true, uint32(hospitalList.length), REP_MAX, 0, 0, 0);
        hospitalList.push(h);
        activeCount += 1;
        emit HospitalRegistered(h, name);
    }

    function reactivate(address h) external onlyGovernor {
        require(hospitals[h].registered, "unknown");
        if (!hospitals[h].active) activeCount += 1;
        hospitals[h].active = true;
        hospitals[h].reputation = minReputation;
    }

    /// Anyone may open the next round once the previous one is closed.
    function startRound() external {
        if (currentRound > 0) {
            Phase p = rounds[currentRound].phase;
            require(p == Phase.Finalized || p == Phase.Failed, "round open");
        }
        currentRound += 1;
        Round storage r = rounds[currentRound];
        r.commitEnd = uint64(block.number) + commitPeriod;
        r.revealEnd = r.commitEnd + revealPeriod;
        r.voteEnd = r.revealEnd + votePeriod;
        r.phase = Phase.Commit;
        r.prevGlobal = latestGlobal;
        require(activeCount > 0, "no hospitals");
        r.quorum = activeCount - (activeCount - 1) / 3;
        emit RoundStarted(currentRound, r.commitEnd, r.revealEnd, r.voteEnd);
    }

    // ------------------------------------------------------------- protocol
    function commitmentOf(bytes32 updateHash, bytes32 uri, uint32 epsMilli, uint32 weight,
                          bytes32 salt, address sender, uint256 round) public view returns (bytes32) {
        return keccak256(abi.encode(updateHash, uri, epsMilli, weight, salt, sender, round,
                                    block.chainid, address(this)));
    }

    function commit(bytes32 c) external payable onlyActive {
        Round storage r = rounds[currentRound];
        require(r.phase == Phase.Commit && block.number <= r.commitEnd, "not commit phase");
        require(commitments[currentRound][msg.sender] == bytes32(0), "committed");
        require(msg.value == deposit, "deposit");
        commitments[currentRound][msg.sender] = c;
        r.nCommit += 1;
        emit Committed(currentRound, msg.sender, c);
    }

    function reveal(bytes32 updateHash, bytes32 uri, uint32 epsMilli, uint32 weight, bytes32 salt) external {
        Round storage r = rounds[currentRound];
        require(block.number > r.commitEnd && block.number <= r.revealEnd, "not reveal phase");
        if (r.phase == Phase.Commit) r.phase = Phase.Reveal;
        bytes32 c = commitments[currentRound][msg.sender];
        require(c != bytes32(0), "no commitment");
        RevealData storage rv = reveals[currentRound][msg.sender];
        require(!rv.revealed, "revealed");
        require(commitmentOf(updateHash, uri, epsMilli, weight, salt, msg.sender, currentRound) == c, "binding");
        uint32 order = uint32(revealOrder[currentRound].length);
        reveals[currentRound][msg.sender] = RevealData(updateHash, uri, epsMilli, weight, order, true);
        revealOrder[currentRound].push(msg.sender);
        r.nReveal += 1;
        emit Revealed(currentRound, msg.sender, updateHash, uri, epsMilli, weight, order, salt);
    }

    function decisionOf(bytes32 globalHash, uint256 keptBitmap, uint32 thresholdMilli) public pure returns (bytes32) {
        return keccak256(abi.encode(globalHash, keptBitmap, thresholdMilli));
    }

    function vote(bytes32 globalHash, uint256 keptBitmap, uint32 thresholdMilli) external {
        Round storage r = rounds[currentRound];
        require(block.number > r.revealEnd && block.number <= r.voteEnd, "not vote phase");
        require(r.phase == Phase.Reveal || r.phase == Phase.Vote || r.phase == Phase.Commit, "closed");
        if (r.phase != Phase.Vote) r.phase = Phase.Vote;
        require(reveals[currentRound][msg.sender].revealed, "only revealers vote");
        require(votes[currentRound][msg.sender] == bytes32(0), "voted");
        bytes32 d = decisionOf(globalHash, keptBitmap, thresholdMilli);
        votes[currentRound][msg.sender] = d;
        tally[currentRound][d] += 1;          // one hospital, one vote
        emit Voted(currentRound, msg.sender, d, 1);
        // BFT quorum q = N - floor((N-1)/3) over hospitals active at round start
        if (tally[currentRound][d] >= r.quorum) {
            r.phase = Phase.Finalized;
            r.decision = d;
            r.globalHash = globalHash;
            r.keptBitmap = keptBitmap;
            r.thresholdMilli = thresholdMilli;
            latestGlobal = globalHash;
            emit Finalized(currentRound, globalHash, keptBitmap, thresholdMilli);
        }
    }

    /// Closes a round that missed its vote deadline (liveness fallback).
    function closeExpired() external {
        Round storage r = rounds[currentRound];
        require(r.phase != Phase.Finalized && r.phase != Phase.Failed, "closed");
        require(block.number > r.voteEnd, "not expired");
        r.phase = Phase.Failed;
        emit RoundFailed(currentRound);
    }

    /// Applies deposits and reputation for one hospital in a closed round.
    function settle(uint256 round, address h) public {
        Round storage r = rounds[round];
        require(r.phase == Phase.Finalized || r.phase == Phase.Failed, "open");
        require(!settled[round][h], "settled");
        settled[round][h] = true;
        Hospital storage hs = hospitals[h];
        bytes32 c = commitments[round][h];
        if (c == bytes32(0)) { emit Settled(round, h, 0, false); return; }
        RevealData storage rv = reveals[round][h];
        int256 delta;
        bool slashed;
        if (!rv.revealed) {
            delta = -15;
            slashed = true;
            hs.missedReveals += 1;
            slashedPool += deposit;
        } else {
            hs.rounds += 1;
            if (r.phase == Phase.Finalized && (r.keptBitmap >> rv.order) & 1 == 0) {
                delta = -5;
                hs.excluded += 1;
            } else if (r.phase == Phase.Finalized) {
                delta = 5;
            }
            (bool ok, ) = payable(h).call{value: deposit}("");
            require(ok, "refund");
        }
        int256 rep = int256(uint256(hs.reputation)) + delta;
        if (rep < 0) rep = 0;
        if (rep > int256(uint256(REP_MAX))) rep = int256(uint256(REP_MAX));
        hs.reputation = uint64(uint256(rep));
        if (hs.reputation < minReputation && hs.active) {
            hs.active = false;
            activeCount -= 1;
        }
        emit Settled(round, h, delta, slashed);
    }

    function settleAll(uint256 round) external {
        for (uint256 i = 0; i < hospitalList.length; i++) {
            if (!settled[round][hospitalList[i]]) settle(round, hospitalList[i]);
        }
    }

    // ------------------------------------------------------------------ views
    function revealersOf(uint256 round) external view returns (address[] memory) { return revealOrder[round]; }
    function hospitalCount() external view returns (uint256) { return hospitalList.length; }
}
