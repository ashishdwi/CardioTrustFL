const solc = require('solc');
const fs = require('fs');
const path = require('path');
const here = __dirname;
const src = fs.readFileSync(path.join(here, 'CardioTrustFL.sol'), 'utf8');
const input = {language:'Solidity', sources:{'CardioTrustFL.sol':{content:src}},
  settings:{optimizer:{enabled:true, runs:200}, viaIR:true, outputSelection:{'*':{'*':['abi','evm.bytecode.object']}}}};
const out = JSON.parse(solc.compile(JSON.stringify(input)));
(out.errors||[]).forEach(e=>console.error(e.formattedMessage));
const c = out.contracts['CardioTrustFL.sol']['CardioTrustFL'];
fs.writeFileSync(path.join(here, 'CardioTrustFL.json'), JSON.stringify({abi:c.abi, bytecode:c.evm.bytecode.object}));
console.log('bytecode bytes', c.evm.bytecode.object.length/2);
