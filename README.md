# Network Intrusion Detection System Using OneClassSVM

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-stable-brightgreen)

A comprehensive machine learning solution for network intrusion detection using OneClassSVM on the NF-CSE-CIC-IDS2018-v2 dataset. This project implements an end-to-end pipeline for anomaly-based network intrusion detection, from data preprocessing to model optimization and evaluation.

## 📋 Table of Contents

- [Network Intrusion Detection System Using OneClassSVM](#network-intrusion-detection-system-using-oneclasssvm)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔍 Project Overview](#-project-overview)
  - [📊 Dataset Description](#-dataset-description)
    - [Dataset Statistics](#dataset-statistics)
    - [Working Dataset](#working-dataset)
  - [✨ Features](#-features)
    - [Feature Descriptions](#feature-descriptions)
  - [📄 License](#-license)

## 🔍 Project Overview

This project implements a machine learning approach to network intrusion detection using the One-Class Support Vector Machine (OneClassSVM) algorithm. The system is designed to learn the characteristics of normal network traffic and identify anomalies that may represent potential security threats or intrusions.

The complete pipeline includes:

- Data preprocessing tailored for network flow data
- Feature engineering for IP addresses and network traffic attributes
- Exploratory data analysis with dimensionality reduction
- Model training with hyperparameter optimization
- Custom threshold optimization for anomaly detection
- Comprehensive performance evaluation and visualization
- Model interpretation techniques

## 📊 Dataset Description

The project uses the NF-CSE-CIC-IDS2018-v2 dataset, a NetFlow-based variant derived from the original CSE-CIC-IDS2018 packet capture files. This dataset is particularly valuable for network security research as it contains real network traffic with labeled normal and attack flows.

### Dataset Statistics

- Total flows: 18,893,708
- Attack samples: 2,258,141 (11.95%)
- Benign samples: 16,635,567 (88.05%)

| Class        | Count      | Description                                                      |
| ------------ | ---------- | ---------------------------------------------------------------- |
| Benign       | 16,635,567 | Normal unmalicious flows                                         |
| BruteForce   | 120,912    | Attempts to obtain credentials through exhaustive trial          |
| Bot          | 143,097    | Remote control of hijacked computers for malicious activities    |
| DoS          | 483,999    | Denial of Service attacks to overload resources                  |
| DDoS         | 1,390,270  | Distributed Denial of Service from multiple sources              |
| Infiltration | 116,361    | Inside attacks that exploit applications through malicious files |
| Web Attacks  | 3,502      | Including SQL injections, command injections, etc.               |

The dataset consists of 46 features representing various characteristics of network flows, including:

- IP addresses and port numbers
- Protocol information
- Flow duration and packet statistics
- Traffic volume metrics
- TCP flag information
- Packet size distributions

### Working Dataset

For this project, we work with a 1% sample of the full dataset, resulting in 188,938 flows with 46 features. Even in this sampled dataset, the class imbalance is preserved:

Normal (Label 0): 166,356 flows (88.05%)
Attack (Label 1): 22,582 flows (11.95%)

## ✨ Features

The dataset contains 46 features representing various aspects of network flows. Key feature categories include:

- Connection identifiers: IP addresses, port numbers, protocols
- Volume metrics: Bytes and packets in both directions
- Timing information: Flow durations, throughput
- Packet characteristics: Sizes, TTL values
- Protocol-specific fields: TCP flags, DNS and ICMP information

### Feature Descriptions

| Feature Index | Feature Name                | Description                                         |
| ------------- | --------------------------- | --------------------------------------------------- |
| 0             | IPV4_SRC_ADDR               | IPv4 source address                                 |
| 1             | IPV4_DST_ADDR               | IPv4 destination address                            |
| 2             | L4_SRC_PORT                 | IPv4 source port number                             |
| 3             | L4_DST_PORT                 | IPv4 destination port number                        |
| 4             | PROTOCOL                    | IP protocol identifier byte                         |
| 5             | L7_PROTO                    | Layer 7 protocol (numeric)                          |
| 6             | IN_BYTES                    | Incoming number of bytes                            |
| 7             | OUT_BYTES                   | Outgoing number of bytes                            |
| 8             | IN_PKTS                     | Incoming number of packets                          |
| 9             | OUT_PKTS                    | Outgoing number of packets                          |
| 10            | FLOW_DURATION_MILLISECONDS  | Flow duration in milliseconds                       |
| 11            | TCP_FLAGS                   | Cumulative of all TCP flags                         |
| 12            | CLIENT_TCP_FLAGS            | Cumulative of all client TCP flags                  |
| 13            | SERVER_TCP_FLAGS            | Cumulative of all server TCP flags                  |
| 14            | DURATION_IN                 | Client to Server stream duration (msec)             |
| 15            | DURATION_OUT                | Client to Server stream duration (msec)             |
| 16            | MIN_TTL                     | Min flow TTL                                        |
| 17            | MAX_TTL                     | Max flow TTL                                        |
| 18            | LONGEST_FLOW_PKT            | Longest packet (bytes) of the flow                  |
| 19            | SHORTEST_FLOW_PKT           | Shortest packet (bytes) of the flow                 |
| 20            | MIN_IP_PKT_LEN              | Len of the smallest flow IP packet observed         |
| 21            | MAX_IP_PKT_LEN              | Len of the largest flow IP packet observed          |
| 22            | SRC_TO_DST_SECOND_BYTES     | Src to dst Bytes/sec                                |
| 23            | DST_TO_SRC_SECOND_BYTES     | Dst to src Bytes/sec                                |
| 24            | RETRANSMITTED_IN_BYTES      | Number of retransmitted TCP flow bytes (src->dst)   |
| 25            | RETRANSMITTED_IN_PKTS       | Number of retransmitted TCP flow packets (src->dst) |
| 26            | RETRANSMITTED_OUT_BYTES     | Number of retransmitted TCP flow bytes (dst->src)   |
| 27            | RETRANSMITTED_OUT_PKTS      | Number of retransmitted TCP flow packets (dst->src) |
| 28            | SRC_TO_DST_AVG_THROUGHPUT   | Src to dst average throughput (bps)                 |
| 29            | DST_TO_SRC_AVG_THROUGHPUT   | Dst to src average throughput (bps)                 |
| 30            | NUM_PKTS_UP_TO_128_BYTES    | Packets whose IP size <= 128                        |
| 31            | NUM_PKTS_128_TO_256_BYTES   | Packets whose IP size > 128 and <= 256              |
| 32            | NUM_PKTS_256_TO_512_BYTES   | Packets whose IP size > 256 and <= 512              |
| 33            | NUM_PKTS_512_TO_1024_BYTES  | Packets whose IP size > 512 and <= 1024             |
| 34            | NUM_PKTS_1024_TO_1514_BYTES | Packets whose IP size > 1024 and <= 1514            |
| 35            | TCP_WIN_MAX_IN              | Max TCP Window (src->dst)                           |
| 36            | TCP_WIN_MAX_OUT             | Max TCP Window (dst->src)                           |
| 37            | ICMP_TYPE                   | ICMP Type * 256 + ICMP code                         |
| 38            | ICMP_IPV4_TYPE              | ICMP Type                                           |
| 39            | DNS_QUERY_ID                | DNS query transaction Id                            |
| 40            | DNS_QUERY_TYPE              | DNS query type (e.g. 1=A, 2=NS..)                   |
| 41            | DNS_TTL_ANSWER              | TTL of the first A record (if any)                  |
| 42            | FTP_COMMAND_RET_CODE        | FTP client command return code                      |
| 43            | Label                       | Binary classification (0=Normal, 1=Attack)          |
| 44            | Attack                      | Specific attack name if applicable                  |
| 45            | Attack_Category             | Category of attack if applicable                    |


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

© 2025 [Ahmed Gamal Okka]. All Rights Reserved.
