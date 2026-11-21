# Data

Both files contain records from the **UCI Heart Disease** collection (Janosi, Steinbrunn,
Pfisterer and Detrano, 1988; https://doi.org/10.24432/C52P4X), which the UCI Machine Learning
Repository distributes under the Creative Commons Attribution 4.0 (CC BY 4.0) licence.
The records are de-identified; no new data were collected.

| File | Contents |
|---|---|
| `heart_combined_1190.csv` | Widely distributed combined file (1,190 records, 11 shared attributes): Cleveland, Hungary, Switzerland, VA Long Beach and Statlog blocks in sequence |
| `cleveland_303.csv` | The separately distributed 303-patient Cleveland file, used only to verify the Cleveland block boundaries |

`ctfl/data.py` recovers the four institutional sites from the combined file, checks them against
site-specific signatures, excludes the Statlog block (237 of its 270 records duplicate Cleveland
records), and builds the federated scenarios used in the paper.

If you use the data, please cite:
Detrano R. et al. (1989) International application of a new probability algorithm for the
diagnosis of coronary artery disease. American Journal of Cardiology 64(5):304–310; and
Janosi A., Steinbrunn W., Pfisterer M., Detrano R. (1988) Heart Disease. UCI Machine Learning
Repository. https://doi.org/10.24432/C52P4X
