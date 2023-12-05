
# NovelQSI

This is a github repository of my research project. You can find here all the important code that I've used. 

- [TriviaQA_to_SQuAD.ipynb](https://github.com/Kkordik/NovelQSI/blob/main/TriviaQA_to_SQuAD.ipynb) contains notebook python code for installing and converting original TriviaQA dataset to the dataset in the JSON format:

    ```
    {"id": "<id of the row>", "context": "context text", "question": "question text", "answers": {"text": ["answer1"], "answer_start": [<id where answer1 starts in the context>]}}
    ```

- [Compare_Models.ipynb](https://github.com/Kkordik/NovelQSI/blob/main/Compare_Models.ipynb) contains notebook python code for comparing models. It was used for two base models comapring and for chosen base model with fine-tuned model comapring.

- [reformat_dialogism.py](https://github.com/Kkordik/NovelQSI/blob/main/reformat_dialogism.py) contains python code for reformating [project-dialogism-novel-corpus](https://github.com/Priya22/project-dialogism-novel-corpus/tree/master) dataset to previously mentioned JSON format
    
     Where "context" structure is: 
    
    ```
    {"context": """
    Characters:
    <characters description>
    Summary:
    <novel summary till the current context window>
    Novel Text:
    <context window with the quote in the middle>"""}
    ```
    The "question" structure is:
    ```
    {"question": """Who said "<the quote>"?"""}
    ```
    
    And the "answer" structure is:
    ```
    {"answer": {"text":["<one of the names from the characters description>"], "answer_start": [<id where the name starts in the context>]}}
    ```

    The "id" left without important changes.

- [/TheGambler](https://github.com/Kkordik/NovelQSI/tree/main/TheGambler) is a part of the [project-dialogism-novel-corpus](https://github.com/Priya22/project-dialogism-novel-corpus/tree/master) dataset, that was reformatted and used for fine-tuning.

- [train_Longformer.ipynb](https://github.com/Kkordik/NovelQSI/blob/main/train_Longformer.ipynb) contains notebook python code for fine-tuning chosen base model. Running the notebook requires A100 accelerator in the Google Colaboratory.
