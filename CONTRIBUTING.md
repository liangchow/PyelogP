# Contributing

Thank you for checking out the `PyelogP` repository and for your interest in contributing to the project. 

The current implementation follows the strain-energy approach (Becker et al., 1987) as closely as possible based on our interpretation of the method. As practicing geotechnical engineers, my co-author and I aim to provide a tool that is useful to the geotechnical community, informed by both practical experience and judgment. However, the codebase is still evolving and would benefit from further refinement, validation, and improvements.

We welcome contributions in the following areas:

- Validate using additional datasets
- Extend validation across constant rate of strain (CRS) oedometer testing results
- Expand documentation, examples, and tutorials, i.e., Jupyter Notebook
- Write unit tests and improving test coverage
- Submit a bug report or feature request on [GitHub Issues](https://github.com/liangchow/PyelogP/issues)

## Quick Start

1. Fork the repository into your own GitHub account. Click **Fork** (top-right). This creates a copy under your account.
2. Clone your fork locally.
    ```bash
    git clone https://github.com/YOUR_USERNAME/PyelogP.git
    cd PyelogP
    ```
3. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate      # macOS/Linux
    .venv\Scripts\activate         # Windows
    ```
4. (Optional but recommended) Upgrade pip:
    ```bash
    pip install --upgrade pip
    ```
5. Install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Before Creating a Pull Request (PR)

You can [create a PR](https://github.com/liangchow/PyelogP/pulls) to suggest modifications regarding your task.

- Discuss or provide a summary of your task.
- If you have used other resources from external sites, provide references in your PR.
