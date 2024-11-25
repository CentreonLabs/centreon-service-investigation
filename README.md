# CSI - Centreon Service Investigation

This CLI tool provides functionalities to compute pairwise distances and perform clustering on service data based on status sequences or just service names. It uses the Levenshtein distance algorithm by default and DBSCAN clustering.

# Features

- **Distance Calculation:** Compute pairwise distances for a given ID and return other IDs below a threshold, sorted by closest to furthest.
- **Clustering:** Perform clustering using DBSCAN based on the chosen distance algorithm.

# Installation

The easiest way to install this tool is with `pipx`. If you don't have it installed, you can do so with the following command:

```bash
# On Ubuntu
sudo apt update && sudo apt install pipx && pipx ensurepath

# On Fedora
sudo dnf install pipx && pipx ensurepath

# On macOS
brew install pipx

# On Windows
# Have python
py -m pip install --user pipx
py -m pipx ensurepath
# restart terminal
```

Then, you can install the tool with the following command:

```bash
pipx install git+https://github.com/centreonlabs/centreon-service-investigation.git
```

# Usage

## Prerequisites

Before running the tool, you need to have a CSV file containing the service data. The file should have the following columns:

- `ctime`: Timestamp of the status change.
- `id`: Name of the service.
- `status`: Status of the service at a given time (as integer).

If you have a Centreon instance, you can export the data from the database with the following query:

```sql
SELECT ctime, CONCAT(host_name, ':', service_description) as id, status 
FROM logs 
WHERE host_name != '' AND service_description != ''
INTO OUTFILE 'centreon_statuses.csv' 
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"' 
LINES TERMINATED BY '\n';
```

## Distance Calculation

To compute pairwise distances for a given ID:

```bash
csi distance <file> <id> [options]
```

### Options

`--compare-names`: Compare names instead of status sequences.
`--threshold, -t`: Distance threshold (default: infinity).
`--algorithm, -a`: Distance algorithm (default: levenshtein).

### Example

```bash
csi distance data/centreon_statuses.csv 'host1:service1' --threshold 5
```

## Clustering

To perform clustering using DBSCAN:

```bash
csi cluster <file> [options]
```

### Options

`--compare-names:` Compare names instead of status sequences.
`--eps:` DBSCAN eps value (default: 3).
`--min_samples:` DBSCAN min_samples value (default: 2).
`--algorithm:` Distance algorithm (default: levenshtein).

### Example

```bash
csi cluster data/centreon_statuses.csv --eps 5
```
