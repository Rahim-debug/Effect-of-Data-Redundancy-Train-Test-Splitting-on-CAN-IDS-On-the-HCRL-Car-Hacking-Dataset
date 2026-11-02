# Decision log (date, decision, reason)
- Bytes beyond the DLC are stored as 0; DLC is part of the signature so nothing is ambiguous.
- Signature = (id, dlc, d0..d7); timestamp excluded.

# Decision log (01/11/2026, decision, reason)

Bytes beyond the DLC are stored as 0. A frame with DLC=2 has only two real bytes, but the model needs 8 feature columns. Because dlc is part of the signature, a frame [05 21] with DLC 2 is never confused with [05 21 00 00 00 00 00 00] with DLC 8.
Signature = (id, dlc, d0…d7). It leaves out the timestamp and the label.
Timestamp is excluded because it is different for every frame, so including it would make every frame unique and hide all the redundancy.
Label is excluded because the same signature can appear under two labels, for example a replayed normal frame inside an attack. The audit measures this (ρc, κ) on days 2-3, which only works if the signature ignores the label.

                  min            max
source                              
dos    1478198376.389 1478201209.059
fuzzy  1478195721.904 1478201209.059
gear   1478193190.057 1478201209.059
normal 1479121434.850 1479121941.287
rpm    1478191030.045 1478201209.059

data\raw\Fuzzy_dataset.csv:1546677:1478196832.714282,0105,6,eb,01,b7,00,98,02,R