---
title: "Creating a Chatbot with GPT-3"
description: "A step-by-step guide to building a chatbot with GPT-3 using the OpenAI API."
date: 2023-04-16T00:33:47-03:00
images: ["/images/articles/creating-a-chatbot-gpt3/gpt-3-thumb.png"]
tags: ["ai", "python", "openai", "chatbot", "fine-tuning"]
authors:
  - name: Felipe Ferreira
    link: https://github.com/felipesilfer
---

![](/images/articles/creating-a-chatbot-gpt3/gpt-3-thumb.png "OpenAI GPT-3")

## Motivation
In 2022, OpenAI released ChatGPT, a chatbot based on GPT-3.5, an improved version of the GPT-3 language model.

GPT-3 has more than 175 billion parameters. OpenAI offers a public API to build applications with it, such as text completion or image generation, and gives you free credit to try it out.

## The project

I decided to use the OpenAI API to build a chatbot with a simple use case: answering students at a generic university.

The idea is for the chatbot to answer questions about the university, such as contacts, address, available courses and procedures.

In this post you will learn how to:

1. Set up access to the OpenAI GPT-3 API;
2. Build a general-purpose chatbot using the GPT-3 Davinci model;
3. Prepare the data to fine-tune GPT-3;
4. Build a university help desk chatbot with the trained model.

## What you will need

In this post I assume you:

* Have basic Python knowledge.
* Already have Python 3 installed.
* Are on Linux (or Windows with WSL2).

Notes:

* This post is adapted from a Jupyter Notebook I made. I plan to publish the repository in the future.
* You may need to make some adjustments. 😉
* The code comments and the chatbot's sample outputs are in Portuguese, since the chatbot was built for Brazilian Portuguese.

---

## Step 1: Create an OpenAI account

Go to the [OpenAI Sign Up page](https://platform.openai.com/signup) and follow the instructions to create an account.

![](/images/articles/creating-a-chatbot-gpt3/signup-opneai.png)

## Step 2: Create an API key

After logging into your OpenAI account, click **Personal** in the top right corner, then **View API keys**.

![](/images/articles/creating-a-chatbot-gpt3/create-api-key.png)

On the **API Keys** page, click **+ Create new secret key** to generate the key we will use in the code.

![](/images/articles/creating-a-chatbot-gpt3/create-new-secret-key.png)

Save the generated key in a YAML file named ``openai-apikey.yml`` so the code can read it. The file looks like this:

```YAML
API-key:  "xx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Step 3: Build a general-purpose chatbot

The code uses the OpenAI and Pandas libraries, so install both:

>Terminal:
```console
pip install openai pandas
```

Also set the ``OPENAI_API_KEY`` environment variable. Add the line below to your shell startup script (for example, `.bashrc` or `.zshrc`) or run it in the terminal before running the code:

>Terminal:
```console
export OPENAI_API_KEY="xx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
With that, the environment is ready to access the API.

Here is a general-purpose chatbot built with GPT-3 through the OpenAI API:

>Python:
```python
import openai, pandas as pd, yaml, json, time

#importando a API-Key
with open('openai-apikey.yml', 'r') as file:
    api_dict = yaml.safe_load(file)

#criando instancia de Chat com a API passando texto de prompt e modelo a ser usado    
def askGPT(text, engine):
    openai.api_key = api_dict['API-key']
    if engine=="text-davinci-003":
        stop = ""
    else:
        stop = "\n"
        
    response = openai.Completion.create(
        engine = engine,
        prompt = text,
        temperature = 0.6,
        max_tokens = 150,
        stop = stop
        
    )
    return print(response.choices[0].text)

#criando interface para o chat
def simple_chat(engine):
    while True:
        print("GPT: Me pergunte algo (digite: #sair para encerrar)\n")
        myQn = input()
        if (myQn=='#sair'):
            break
        askGPT(myQn, engine=engine)
        print('\n')
        
simple_chat("text-davinci-003")
```

```
    GPT: Me pergunte algo (digite: #sair para encerrar)
    
     O que é Jupyter Notebook?

    
    Jupyter Notebook é uma ferramenta open-source que permite a criação e compartilhamento de documentos contendo código, equações, visualizações de dados e texto narrativo. É um ambiente ideal para a computação científica interativa, pois permite a criação de blocos de código reutilizáveis, anotações, visualizações e também a execução de códigos.
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    
     #sair
```


## Step 4: Preparing the dataset for fine-tuning

Fine-tuning adjusts a model so it serves a specific purpose better.

As an exercise, we will fine-tune a chatbot that answers basic questions about a university.

To train the model, we will use the ``intents.json`` dataset, available on Kaggle: [University Chatbot Dataset, by Niralii Vaghani](https://www.kaggle.com/dsv/5024271 "University Chatbot Dataset - by Niralii vaghani").

### Format

According to the documentation, the training dataset shows GPT-3 what you would like it to answer.

The data must be in [JSONL](https://jsonlines.org/) format, where each line is a prompt and completion pair, as in this example:

```JSON
{"prompt": "<prompt text> ->", "completion": " <ideal generated text> \n"}
{"prompt": "<prompt text> ->", "completion": " <ideal generated text> \n"}
{"prompt": "<prompt text> ->", "completion": " <ideal generated text> \n"}
...
```

### Translating and formatting the dataset
The following code translates the dataset from English to Portuguese using the API. It takes a while because of the API limit of 60 requests per minute:

>Python:
```python
#criando instancia de Chat com a API passando o prompt para tradução do dataset   
def translateGPT(text):
    openai.api_key = api_dict['API-key']
    task_description = "Translate English to Brazilian Portuguese:"
    to_translate = text
    prompt = task_description+"\n"+to_translate+" ->"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    response = json.loads(json.dumps(vars(response)))
    time.sleep(1) #a API limita 60 requests / min, necessário 1s de espera por chamada de função
    return response['_previous']['choices'][0]['text']

#reformatando dados do dataset
df = pd.read_json('intents.json')
df_output = pd.DataFrame()
for row in df.index:
    patterns = df.iloc[row].to_dict()['intents']['patterns']
    responses = df.iloc[row].to_dict()['intents']['responses']
    for p in range(0,len(patterns)):
        for r in range(0,len(responses)):
            dict = {}
            dict['prompt'] = translateGPT(patterns[p])+" ->"
            dict['completion'] = " "+translateGPT(responses[r])+".\n"
            df_dict = pd.DataFrame([dict])
            df_output = pd.concat([df_output, df_dict], ignore_index = True)

#salvando dataset formatado para o arquivo 'dataframe.jsonl'            
with open('dataframe.jsonl', 'w', encoding='utf-8') as file:
    file.write(df_output.to_json(orient='records', lines=True, force_ascii=False))
```

To avoid spending the platform's free credit, I translated the original dataset separately and saved the result as `intents_ptBR.json`.

>Python:
```python
#visualizando amostra de dados do dataset orginal:
df = pd.read_json('intents_ptBR.json')
df.head(5)
```

|   |intents|
| :--- | :--- |
|0|{'tag': 'saudação', 'patterns': ['Oi', 'Como v...|
|1|{'tag': 'despedida', 'patterns': ['Tchau', 'At...|
|2|{'tag': 'criador', 'patterns': ['qual é o nome...|
|3|{'tag': 'nome', 'patterns': ['nome', 'seu nome...|
|4|{'tag': 'horario', 'patterns': ['horário da fa...|

</br>

The reformatting code is the same as before, just without the translation through the ``translateGPT()`` function:

>Python:
```python
#reformatando dados do dataset
df_output = pd.DataFrame()
for row in df.index:
    patterns = df.iloc[row].to_dict()['intents']['patterns']
    responses = df.iloc[row].to_dict()['intents']['responses']
    for p in range(0,len(patterns)):
        for r in range(0,len(responses)):
            dict = {}
            dict['prompt'] = patterns[p]+" ->"
            dict['completion'] = " "+responses[r]+".\n"
            df_dict = pd.DataFrame([dict])
            df_output = pd.concat([df_output, df_dict], ignore_index = True)

#visualizamdp amostra do dataset formatado corretamente            
df_output.head(5)
```

|   |prompt|completion|
| :--- | :--- | :--- |
|0|Oi -&gt;|Olá!.\n<|
|1|Oi -&gt;|Bom te ver de novo!.\n|
|2|Oi -&gt;|Olá, como posso ajudar?.\n|
|3|Como você está? -&gt;|Olá!.\n|
|4|Como você está? -&gt;|Bom te ver de novo!.\n|

</br>

Once transformed, the dataset must be saved to a JSONL file, as the documentation requires:

>Python:
```python
#salvando dataset formatado para o arquivo 'dataframe.jsonl'
with open('dataframe.jsonl', 'w', encoding='utf-8') as file:
    file.write(df_output.to_json(orient='records', lines=True, force_ascii=False))
```

### Checking the dataframe quality
The OpenAI library comes with a tool that checks the dataset quality before uploading it for fine-tuning.

The ``openai tools fine_tunes.prepare_data -f`` command analyzes the JSONL file. For our file, we answer ``y`` to remove duplicates, ``n`` to not split the file into training and validation sets (the dataset isn't big enough for that) and ``y`` to save the changes to a new file, ``'dataframe_prepared.jsonl'``. The output looks like this:

>Terminal:
```console
printf "y\nn\ny" | openai tools fine_tunes.prepare_data -f dataframe.jsonl
```

```console
    Analyzing...
    
    - Your file contains 521 prompt-completion pairs
    - Based on your data it seems like you're trying to fine-tune a model for classification
    - For classification, we recommend you try one of the faster and cheaper models, such as `ada`
    - For classification, you can estimate the expected model performance by keeping a held out dataset, which is not used for training
    - There are 32 duplicated prompt-completion sets. These are rows: [66, 67, 68, 69, 89, 152, 156, 160, 190, 196, 211, 218, 233, 244, 256, 296, 368, 369, 372, 394, 395, 396, 397, 418, 459, 463, 480, 481, 490, 491, 496, 497]
    - All prompts end with suffix ` ->`
    
    Based on the analysis we will perform the following actions:
    - [Recommended] Remove 32 duplicate rows [Y/n]: - [Recommended] Would you like to split into training and validation set? [Y/n]: 
    
    Your data will be written to a new JSONL file. Proceed [Y/n]: 
    Wrote modified file to `dataframe_prepared.jsonl`
    Feel free to take a look!
    
    Now use that file when fine-tuning:
    > openai api fine_tunes.create -t "dataframe_prepared.jsonl"
    
    After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string ` ->` for the model to start generating completions, rather than continuing with the prompt. Make sure to include `stop=[".\n"]` so that the generated texts ends at the expected place.
    Once your model starts training, it'll approximately take 14.07 minutes to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.
```

Checking the new dataframe:

>Python:
```python
df = pd.read_json('dataframe_prepared.jsonl', lines=True)
df.head(10)
```

|   |prompt|completion|
| :--- | :--- | :--- |
|0|Oi -&gt;|Olá!.\n<|
|1|Oi -&gt;|Bom te ver de novo!.\n|
|2|Oi -&gt;|Olá, como posso ajudar?.\n|
|3|Como você está? -&gt;|Olá!.\n|
|4|Como você está? -&gt;|Bom te ver de novo!.\n|
|5|Como você está? -&gt;|Olá, como posso ajudar?.\n|
|6|Tem alguém aí? -&gt;|Olá!.\n|
|7|Tem alguém aí? -&gt;|Bom te ver de novo!.\n|
|8|Tem alguém aí? -&gt;|Olá, como posso ajudar?.\n|
|9|Olá -&gt;|Olá!.\n|

</br>


Running the command again, we see the output for a correctly formatted dataframe:

>Terminal:
```console
printf "n" | openai tools fine_tunes.prepare_data -f dataframe_prepared.jsonl
```

```console
    Analyzing...
    
    - Your file contains 489 prompt-completion pairs
    - Based on your data it seems like you're trying to fine-tune a model for classification
    - For classification, we recommend you try one of the faster and cheaper models, such as `ada`
    - For classification, you can estimate the expected model performance by keeping a held out dataset, which is not used for training
    - All prompts end with suffix ` ->`
    
    No remediations found.
    - [Recommended] Would you like to split into training and validation set? [Y/n]: 
    You can use your file for fine-tuning:
    > openai api fine_tunes.create -t "dataframe_prepared.jsonl"
    
    After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string ` ->` for the model to start generating completions, rather than continuing with the prompt. Make sure to include `stop=[".\n"]` so that the generated texts ends at the expected place.
    Once your model starts training, it'll approximately take 14.07 minutes to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.
```

## Step 5: Uploading the dataset for fine-tuning

You can upload the prepared dataframe with the OpenAI command line tool or with the library, as in the example below. The uploaded file gets an ID, which we need to keep to create the fine-tuning:

>Python:
```python
upload_response = openai.File.create(
    file=open('dataframe_prepared.jsonl'),
    purpose='fine-tune'
)
file_id = upload_response.id
upload_response
```

>Terminal:
```console

    <File file id=file-j9xUWUtOagExIj7QiTRmKXbO at 0x7f49aed5bb50> JSON: {
      "bytes": 59864,
      "created_at": 1681604031,
      "filename": "file",
      "id": "file-j9xUWUtOagExIj7QiTRmKXbO",
      "object": "file",
      "purpose": "fine-tune",
      "status": "uploaded",
      "status_details": null
    }
```


### Creating the fine-tuning
After the upload, we train the model with the OpenAI library.

The parameters are ``training_file``, with the ID of the uploaded file, and ``model``, the base model for the fine-tuning, which in our case is ``davinci``.

Keep the fine-tuning ID to follow the training progress:

>Python:
```python
fine_tune = openai.FineTune.create(training_file=file_id, model="davinci")
finetune_id = fine_tune.id
fine_tune
```

```console

    <FineTune fine-tune id=ft-iMXzAEiUcFE4nZLDd5kp6tCY at 0x7f49aedecef0> JSON: {
      "created_at": 1681604044,
      "events": [
        {
          "created_at": 1681604044,
          "level": "info",
          "message": "Created fine-tune: ft-iMXzAEiUcFE4nZLDd5kp6tCY",
          "object": "fine-tune-event"
        }
      ],
      "fine_tuned_model": null,
      "hyperparams": {
        "batch_size": null,
        "learning_rate_multiplier": null,
        "n_epochs": 4,
        "prompt_loss_weight": 0.01
      },
      "id": "ft-iMXzAEiUcFE4nZLDd5kp6tCY",
      "model": "davinci",
      "object": "fine-tune",
      "organization_id": "org-gzduqVnnZJVEbUTxiMIKEE43",
      "result_files": [],
      "status": "pending",
      "training_files": [
        {
          "bytes": 59864,
          "created_at": 1681604031,
          "filename": "file",
          "id": "file-j9xUWUtOagExIj7QiTRmKXbO",
          "object": "file",
          "purpose": "fine-tune",
          "status": "processed",
          "status_details": null
        }
      ],
      "updated_at": 1681604044,
      "validation_files": []
    }
```
Viewing the ID:

>Python:
```python
print(finetune_id)
```

```
ft-iMXzAEiUcFE4nZLDd5kp6tCY
```

To follow the training, use the command line tool with the fine-tuning ID. Run it more than once until it finishes. When training ends, the output shows the ID of the generated model:

>Terminal:
```console
openai api fine_tunes.follow -i ft-iMXzAEiUcFE4nZLDd5kp6tCY
```

```console
    [2023-04-15 21:14:04] Created fine-tune: ft-iMXzAEiUcFE4nZLDd5kp6tCY
    [2023-04-15 21:14:16] Fine-tune costs $2.01
    [2023-04-15 21:14:16] Fine-tune enqueued. Queue number: 0
    [2023-04-15 21:24:19] Fine-tune started
    [2023-04-15 21:28:15] Completed epoch 1/4
    [2023-04-15 21:30:41] Completed epoch 2/4
    [2023-04-15 21:33:06] Completed epoch 3/4
    [2023-04-15 21:35:30] Completed epoch 4/4
    [2023-04-15 21:36:04] Uploaded model: davinci:ft-personal-2023-04-16-00-36-03
    [2023-04-15 21:36:05] Uploaded result file: file-rAuDVNmEI9EaUYuSvvLGtQCZ
    [2023-04-15 21:36:05] Fine-tune succeeded
    
    Job complete! Status: succeeded 🎉
    Try out your fine-tuned model:
    
    openai api completions.create -m davinci:ft-personal-2023-04-16-00-36-03 -p <YOUR_PROMPT>
```

Getting the ``fine_tuned_model`` from the last fine-tuning (check in the previous step that it finished):

>Python:
```python
finetune_list = json.loads(json.dumps(openai.FineTune.list()))
finetuned_model_id = finetune_list['data'][-1]['fine_tuned_model']
print(finetuned_model_id)
```
```
    'davinci:ft-personal-2023-04-16-00-36-03'
```


### Testing the trained model

To test the model, just open a new chat passing ``finetuned_model_id`` as the engine.

This was the test result:

>Python:
```python
simple_chat(finetuned_model_id)
```

```console
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     Olá, com quem eu falo?


     -> Você pode me chamar de Leitor da Mente..
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     Qual o horário de funcionamento da faculdade?


     -> A faculdade está aberta das 8h às 17h, de segunda a sábado!.
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     E tem telefone para contato?


     -> Você pode entrar em contato através do número: (xx) xxxx-xxxx.
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     Quais cursos são disponíveis?


     -> Nossa universidade oferece Tecnologia da Informação, Engenharia da Computação, Engenharia Mecânica, Engenharia Química, Engenharia Civil e Matemática..
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     Tem cantina na faculdade?


     -> Nossa universidade tem uma cantina com uma variedade de alimentos disponíveis..
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     O que tem pra comer na cantina?


     -> Servimos Franky, Locho, Alu-puri, Kachori, Khavsa, Thaali e muitos mais no cardápio..
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     #sair

```


## Conclusion

I managed to build a chatbot with reasonable answers even with a small dataset of about 500 entries. With a larger dataset, the results should be even better.

### Observations

- The OpenAI documentation is a bit lacking: it mixes Python library instructions with HTTP requests in the same place, which makes references hard to find.
- Fine-tuning is cheap (this one cost US$ 2.01), and using the ready-made or trained models costs even less.
- The free trial gives you US$ 18 in credit, which lets you do a lot of cool things at this low cost.
- Most of the effort goes into preparing the dataset, because a poorly built dataset hurts the quality of the answers.

### Future possibilities

In this use case I didn't write more elaborate prompts to improve the answers. You can do that by following the best practices in the documentation, and I recommend it.