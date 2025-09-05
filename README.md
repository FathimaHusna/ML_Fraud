ML_Fraud – Data Ingestion (KYC ID Authenticity)

Contents
- ingestion/: DI client, normalization, storage adapters
- functions/: Decoupled Event Grid consumer (IDP bronze → silver)
- notebooks/: Experimental ingestion notebook
- samples/: Small DI output sample for offline runs

Quickstart
- Local (no Azure):
  1) `pip install -r requirements.txt`
  2) `python ML_Fraud/run_ingest.py --submission-id demo-001 --use-sample`
  3) Inspect outputs under `ML_Fraud/data/bronze` and `ML_Fraud/data/silver`

- Orchestrated (single service; optional)
  1) Create a Document Intelligence resource and set `DI_ENDPOINT`.
  2) Choose storage auth:
     - Connection string: set `AZURE_STORAGE_CONNECTION_STRING` and `STORAGE_MODE=adls` (enables SAS generation).
     - Or Account URL + Managed Identity: set `STORAGE_ACCOUNT_URL` and `STORAGE_MODE=adls` (pre-create SAS or allow public read for DI).
  3) Set `STORAGE_CONTAINER` (default `mlfraud`). Ensure container exists.
  4) Upload and ingest:
     - `python ML_Fraud/run_ingest.py --submission-id cloud-001 --upload-path /path/to/id.jpg`
     - The script uploads to `incoming/<submissionId>/<file>` and calls DI with the SAS/public URL.
  5) Bronze/Silver are written to either local or ADLS depending on `STORAGE_MODE`.

- Decoupled (recommended if separate modules)
  - Assuming the IDP module stores bronze and emits an event with `bronzeUrl` or `bronzePath`:
    - CLI consume bronze directly (no DI call):
      - From URL: `python ML_Fraud/run_ingest.py --submission-id abc-123 --idp-bronze-path "https://<acct>.blob.core.windows.net/<container>/idp/bronze/.../abc-123.json?<sas>"`
      - From container path (with MI): `python ML_Fraud/run_ingest.py --submission-id abc-123 --idp-bronze-path idp/bronze/2025/01/01/abc-123.json`
    - Azure Function consumer (Event Grid): deploy `functions/idp_output_consumer` and subscribe to `idp.output.created`.
      The function downloads bronze, normalizes to silver, and can be extended to compute features/scores.

- Notebook:
  - Open `ML_Fraud/notebooks/01_ingestion_experiment.ipynb` and run cells
