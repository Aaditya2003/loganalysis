# Flipkart_Grid_5.0_InfoSec

## Compliance Monitoring and Enforcement through Log Analysis using Large Language Models

### Project Overview

As Flipkart deals with increasing volumes of data and complex systems, ensuring compliance with security policies, standards, and baselines has become a critical challenge. To address this issue, we propose a project focused on developing a system that leverages large language models for compliance monitoring and enforcement through log analysis from relevant sources.

### Problem Statement

The objective is to build a solution that can effectively analyze logs, system configurations, access controls, and user privileges to check for compliance with security policies and standards. By utilizing the power of large language models (LLMs) like ChatGPT or its open-source alternatives, we aim to automate the process of identifying non-compliant activities and generating actionable insights for remediation.

## Table of Contents

1. [Project Overview](#project-overview)
   - [Problem Statement](#problem-statement)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Usage](#usage)
   - [Configuration](#configuration)
   - [Running the System](#running-the-system)
4. [Features](#features)
5. [Contributing](#contributing)
6. [License](#license)

## Getting Started

### Prerequisites

Before you can run this system, make sure you have the following prerequisites installed:

- Python (>= 3.10)
- Node (>= 18v)
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/darshan8850/Flipkart_Grid_5.0_InfoSec.git

2. Install required libraries and dependencies in seperate python env (prefernce - CONDA)

    ```bash
    pip install -r requirements.txt

3. Traverse to client  (cd client)

    ```bash
    npm install 

4. Run react app 

    ```bash
    npm run start

4. Traverse to main directory 

    ```bash
    python System_generated_Logs/scripts/LLM/main_server.py


## Configuration

Before running the system, you need to configure it to work with your specific environment. The configuration can include defining log sources, security policies, and other parameters. Modify the configuration files in the `config/` directory to match your setup.


## Features

- Automated compliance monitoring and enforcement.
- Log analysis from various sources. (System and Human Generated)
- Customizable configuration for different environments.
- Actionable insights and alerts for non-compliance.
- Integration with large language models for natural language understanding.

## License
This project is licensed under the MIT License.
