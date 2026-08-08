# Attachment kinds exercised by SC-WBD-003

Not every attachment kind the schema declares was exercised: observation, stimulus, boundary_output, context reached no loss.

Of the four kinds `scwbd/schema/attachment.py` declares, 0 reached a loss: none.
The remaining `observation`, `stimulus`, `boundary_output`, `context` did not, and the table below says whether that is because no card declares one or because a declared channel feeds no loss.


| kind | source | channel | state | feeds a loss |
| --- | --- | --- | --- | --- |
| `observation` | `ds000113_real` | `bold` | disabled | no |
| `observation` | `ds000117_real` | `eeg` | admitted_not_yet_measured | yes |
| `observation` | `ds002336_real` | `eeg` | admitted_not_yet_measured | yes |
| `observation` | `ds002336_real` | `bold` | admitted_not_yet_measured | yes |
| `observation` | `ds004024_perturb` | `eeg` | admitted_not_yet_measured | yes |
| `observation` | `ds004024_rest_real` | `eeg` | admitted_not_yet_measured | yes |
| `observation` | `sleepedf_real` | `eeg_bipolar` | admitted_not_yet_measured | yes |
| `stimulus` | `ds000117_real` | `face_stimulus` | declared_only | no |
| `stimulus` | `ds004024_perturb` | `tms_pulse` | admitted_not_yet_measured | yes |
| `boundary_output` | `ds000113_real` | `cardiac_respiratory` | disabled | no |
| `boundary_output` | `ds000113_real` | `eyegaze` | disabled | no |
| `boundary_output` | `ds000117_behaviour` | `button_press` | admitted_not_yet_measured | yes |
| `boundary_output` | `ds000117_behaviour` | `response_time` | admitted_not_yet_measured | yes |
| `boundary_output` | `ds004024_perturb` | `emg` | declared_only | no |
| `boundary_output` | `ds004024_rest_real` | `emg` | declared_only | no |
| `boundary_output` | `ds004024_rest_real` | `ecg` | declared_only | no |
| `boundary_output` | `ds004024_rest_real` | `eog` | declared_only | no |
| `context` | — | — | **no card declares one** | no |

## Why this table is not the contributed-gradient table

A run can have every enabled source contribute gradient while three of the four attachment kinds are untouched, because a source list can be entirely observations. Only the per-kind answer speaks to the schematic that gives boundary_output equal billing with observation.
