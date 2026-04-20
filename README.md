# MIMIC Clinical Decision Making Framework - Hyperscaler AI Support


## Overview

This is a fork of Paul Hager's [MIMIC Clinical Decision Making Framework](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Framework), adding support for running the MIMIC-CDM task using any of:

- Google Vertex AI (Gemini 2.5 Flash Lite, Gemini 3.1 Flash Lite Preview, Gemini 3.1 Pro Preview)
- Microsoft Azure (Open AI's GPT 5.4)
- Amazon Bedrock (Anthropic Claude Sonnet 4.6, Claude Opus 4.6)

The original README for this repo is [here](README-original.md).


## Important!

In order to ensure compliance with [MIMIC data usage and licensing terms](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/),
in particular [regarding the use of LLMs](https://physionet.org/news/post/llm-responsible-use/), it is
**absolutely necessary** to run tasks in an environment in which 1) zero data retention, 2) no use of data for model training, and 3) no human review can be ensured. At the time of writing (Apr 2026), this can be done in the following way for each provider:

- Microsoft Azure Foundry:
  - Documentation: [Data, privacy, and security for Azure Direct Models in Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/openai/data-privacy)
  - As noted in the documentation, asynchronous abuse monitoring results in 30-day data retention, violating ZDR. A request for modified abuse monitoring may be applied for [here](https://customervoice.microsoft.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbR7en2Ais5pxKtso_Pz4b1_xUOE9MUTFMUlpBNk5IQlZWWkcyUEpWWEhGOCQlQCN0PWcu). Use of Azure Foundry should not be attempted without approval from Microsoft of modified abuse monitoring.
  - As explained in the documentation, confirm that the value of `"ContentLogging"` in the approved subscription's Capabilities list is set to `"false"`.
- Google Vertex AI:
  - Documentation: [Vertex AI and zero data retention](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/vertex-ai-zero-data-retention)
  - Apply for exclusion from Abuse Monitoring [here](https://forms.gle/mtjKKas8a82grYN6A). Upon acceptance, abuse monitoring may be suspended for the particular project id.
  - Avoid the use of session resumption as noted in the documentation. Grounding with Google and Grounding with Google are obviously not used.
  - Caching should also be disabled in accordance with the instructions [here](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/vertex-ai-zero-data-retention#enabling-disabling-caching).
- Amazon Bedrock:
  - Documentation:
    - [Data protection](https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html)
	- [Amazon Bedrock abuse detection](https://docs.aws.amazon.com/bedrock/latest/userguide/abuse-detection.html)
	  - "There is no human review of, or access to, user inputs or model outputs."
	  - "Amazon Bedrock does not store user input or model output and does not share these with third-party model providers."
	- [Amazon Bedrock FAQs](https://aws.amazon.com/bedrock/faqs/)
	  - Q: "Are user inputs and model outputs made available to third party model providers?"
	  - A: "No. Users' inputs and model outputs are not shared with any model providers."
	  - Q: "What security and compliance standards does Amazon Bedrock support?"
	  - A: "With Amazon Bedrock, your content is not used to improve the base models and is not shared with any model providers."
	  - Q: "Will AWS and third-party model providers use customer inputs to or outputs from Amazon Bedrock to train Amazon Nova, Amazon Titan or any third-party models?"
	  - A: "No, AWS and the third-party model providers will not use any inputs to or outputs from Amazon Bedrock to train Amazon Nova, Amazon Titan, or any third-party models."


## Environment

To setup the environment, create a new virtual environment of your choosing with python=3.10, install
the libraries from requirements.txt, and install other dependencies:

```
pip install --no-deps -r requirements.txt
python -m nltk.downloader punkt stopwords
```

Note that in order to simplify running just the cloud models and minimize updates necessary for the
older version of Python in the original repository, requirements have been simplified to only include
requirements necessary to run the cloud models. For running new local models, please refer to the
[original repository](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Framework).

# Citation

If you found this code and dataset useful, please cite the MIMIC-CDM paper and dataset with:

Hager, P., Jungmann, F., Holland, R. et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making. Nat Med (2024). https://doi.org/10.1038/s41591-024-03097-1
```
@article{hager_evaluation_2024,
	title = {Evaluation and mitigation of the limitations of large language models in clinical decision-making},
	issn = {1546-170X},
	url = {https://doi.org/10.1038/s41591-024-03097-1},
	doi = {10.1038/s41591-024-03097-1},,
	journaltitle = {Nature Medicine},
	shortjournal = {Nature Medicine},
	author = {Hager, Paul and Jungmann, Friederike and Holland, Robbie and Bhagat, Kunal and Hubrecht, Inga and Knauer, Manuel and Vielhauer, Jakob and Makowski, Marcus and Braren, Rickmer and Kaissis, Georgios and Rueckert, Daniel},
	date = {2024-07-04},
}
```

Hager, P., Jungmann, F., & Rueckert, D. (2024). MIMIC-IV-Ext Clinical Decision Making: A MIMIC-IV Derived Dataset for Evaluation of Large Language Models on the Task of Clinical Decision Making for Abdominal Pathologies (version 1.0). PhysioNet. https://doi.org/10.13026/2pfq-5b68.
```
@misc{hager_mimic-iv-ext_nodate,
	title = {{MIMIC}-{IV}-Ext Clinical Decision Making: A {MIMIC}-{IV} Derived Dataset for Evaluation of Large Language Models on the Task of Clinical Decision Making for Abdominal Pathologies},
	url = {https://physionet.org/content/mimic-iv-ext-cdm/1.0/},
	shorttitle = {{MIMIC}-{IV}-Ext Clinical Decision Making},
	publisher = {{PhysioNet}},
	author = {Hager, Paul and Jungmann, Friederike and Rueckert, Daniel},
	urldate = {2024-07-04},
	doi = {10.13026/2PFQ-5B68},
	note = {Version Number: 1.0
Type: dataset},
}
```
