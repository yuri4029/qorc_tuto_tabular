# Data

The tutorial expects `data/NSL_KDD_labeled.csv`: the NSL-KDD `KDDTrain+` split
(125,973 rows) with a header row, the 41 feature columns and a `label` column
holding `normal` or the attack name.

The dataset is published by the Canadian Institute for Cybersecurity:
<https://www.unb.ca/cic/datasets/nsl.html>. The original `KDDTrain+.txt` has no
header and two extra trailing columns (the attack label and a difficulty
score). To convert it:

```python
import pandas as pd

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
df = pd.read_csv("KDDTrain+.txt", header=None, names=columns).drop(columns="difficulty")
df.to_csv("data/NSL_KDD_labeled.csv", index=False)
```

Data files in this folder are git-ignored.
