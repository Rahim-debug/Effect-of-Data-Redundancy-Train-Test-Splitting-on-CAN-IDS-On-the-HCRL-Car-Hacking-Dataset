# Decision log (date, decision, reason)
- Bytes beyond the DLC are stored as 0; DLC is part of the signature so nothing is ambiguous.
- Signature = (id, dlc, d0..d7); timestamp excluded.
