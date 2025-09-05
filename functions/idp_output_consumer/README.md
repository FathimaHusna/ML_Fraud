IDP Output Consumer (Decoupled)

Purpose
- Consume `idp.output.created` events, fetch the bronze JSON, normalize (silver), compute features, score, persist case, and emit `fraud.score.created`.

Event Contract (example)
```
{
  "eventType": "idp.output.created",
  "subject": "/submissions/{submissionId}",
  "eventTime": "2025-01-01T12:00:00Z",
  "data": {
    "submissionId": "abc-123",
    "bronzeUrl": "https://<acct>.blob.core.windows.net/<container>/idp/bronze/2025/01/01/abc-123.json?<sas>",
    "modelVersion": "2023-10-31",
    "docType": "idDocument.passport",
    "captureTs": "2025-01-01T11:59:00Z",
    "contentHash": "sha256:..."
  }
}
```

Notes
- Use Managed Identity for Storage and AML; DI is not called here.
- Prefer bronzeUrl with SAS for simplicity; or emit bronzePath and let the consumer read via MI.

