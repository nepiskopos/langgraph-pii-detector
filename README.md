# Personally Identifiable Information (PII) detector implemented in LangGraph

This repository contains a custom Personally Identifiable Information (PII) detection agent, developed using LangGraph and powered by Azure OpenAI GPT-4o model.

This agent is meant to be combined with the respective pipeline code developed as an enhancement to Open WebUI, and can be found in [this repository](https://github.com/nepiskopos/open-webui-enhancements).

The system is designed to identify sensitive personal data such as names, addresses, phone numbers, emails, and other common PII entities from unstructured text. It uses language model-driven reasoning combined with a graph-based flow for flexible and context-aware processing.

Features

    🔍 PII Detection: Identifies a wide range of PII types using GPT models with natural language reasoning.

    🧠 LangGraph Orchestration: Utilizes LangGraph to define modular, stateful, and extensible detection workflows.

    ☁️ Azure OpenAI Integration: Powered by GPT-4o or other Azure OpenAI models for high-accuracy analysis.

    🛠️ Customizable Pipelines: Easily adaptable for redaction, anonymization, or classification tasks.

    📦 Modular Design: Plug-and-play components to support experimentation or integration into larger systems.

Use Cases

    Preprocessing documents before storage or sharing

    Enhancing data privacy compliance (e.g., GDPR, HIPAA)

    Building secure AI pipelines with privacy-preserving layers

This solution is based on more feature-rich version, developed during my time at [IBM](https://www.ibm.com), as part of the Digital Transformation project for the [National Bank of Greece (NBG)](https://www.nbg.gr/en/), where I was tasked with exploring and utilizing the LangGraph framework to meet complex enterprise needs.
