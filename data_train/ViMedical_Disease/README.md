---
license: cc-by-nc-sa-4.0
configs:
- config_name: default
  data_files:
  - split: train
    path: ViMedical_Disease.csv
task_categories:
- text-classification
- text-generation
- question-answering
- text2text-generation
language:
- vi
tags:
- medical
pretty_name: ViMedical Disease
github: https://github.com/PB3002/ViMedical_Disease
---

This dataset contains over 12K+ questions and symptoms related to various common diseases in Vietnamese. It's designed to aid in the classification of medical symptoms and provide preliminary disease identification. The dataset covers a wide range of diseases, including cardiovascular, digestive, neurological, dermatological, endocrine, and others.

For more information and updates about the dataset, please refer to the **main repository** [here](https://github.com/PB3002/ViMedical_Disease).

## This dataset can be used for:
- Data analysis
- Building disease prediction models
- Creating chatbots
- Providing information to users

### The dataset has two columns:
**`Disease`**:  The name of the disease in Vietnamese.

**`Question`**:  Questions and descriptions of disease symptoms in Vietnamese, often posed as a query seeking information about a possible diagnosis.

## **Important Notes:**
This dataset provides information on disease symptoms, not official medical diagnoses.
Users should consult a doctor for proper diagnosis and treatment.

## 📢 License:
Copyright © 2024 PB. All rights reserved.
This dataset is licensed under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.