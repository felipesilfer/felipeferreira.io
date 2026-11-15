---
title: "Criando um Chatbot com GPT-3"
description: "Um passo a passo para implementar um chatbot com GPT-3 utilizando a OpenAI API."
date: 2023-04-16T00:33:47-03:00
images: ["/images/articles/creating-a-chatbot-gpt3/gpt-3-thumb.png"]
tags: ["ia", "python", "openai", "chatbot", "fine-tuning"]
authors:
  - name: Felipe Ferreira
    link: https://github.com/felipesilfer
---

![](/images/articles/creating-a-chatbot-gpt3/gpt-3-thumb.png "OpenAI GPT-3")

## Motivação
Em 2022, a OpenAI lançou o ChatGPT, um chatbot baseado no GPT-3.5, versão aprimorada do modelo de linguagem GPT-3.

O GPT-3 tem mais de 175 bilhões de parâmetros. A OpenAI oferece uma API pública para criar aplicativos com ele, que completam textos ou geram imagens, por exemplo, e dá um crédito gratuito para testar.

## O projeto

Resolvi usar a API da OpenAI para criar um chatbot com um caso de uso simples: atender estudantes de uma universidade genérica.

A ideia é que o chatbot responda dúvidas sobre a universidade, como contatos, endereço, cursos disponíveis e procedimentos.

Neste artigo você vai aprender a:

1. Configurar o acesso à API do GPT-3 da OpenAI;
2. Criar um chatbot generalista usando o modelo Davinci do GPT-3;
3. Tratar os dados para fazer um fine-tuning do GPT-3;
4. Criar um chatbot de atendimento universitário com o modelo treinado.

## O que será preciso

Neste artigo presumo que você:

* Tem conhecimento básico de Python.
* Já tem o Python 3 instalado.
* Está em ambiente Linux (ou Windows com WSL2).

Observações:

* Este post é uma adaptação de um Jupyter Notebook que eu fiz. Pretendo publicar o repositório no futuro.
* Pode ser que você precise fazer adaptações. 😉

---

## Passo 1: Criar conta OpenAI

Vá até a [página de cadastro (Sign Up) da OpenAI](https://platform.openai.com/signup) e siga as instruções para criar uma conta.

![](/images/articles/creating-a-chatbot-gpt3/signup-opneai.png)

## Passo 2: Criar uma API Key

Depois de entrar na sua conta OpenAI, clique em **Personal**, no canto superior direito, e em seguida em **View API keys**.

![](/images/articles/creating-a-chatbot-gpt3/create-api-key.png)

Na página **API Keys**, clique no botão **+ Create new secret key** para gerar a chave que vamos usar no código.

![](/images/articles/creating-a-chatbot-gpt3/create-new-secret-key.png)

Salve a chave gerada num arquivo YAML chamado ``openai-apikey.yml``, para o código poder lê-la. O arquivo segue este formato:

```YAML
API-key:  "xx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Passo 3: Implementar Chatbot Genérico

O código usa as bibliotecas OpenAI e Pandas, então instale as duas:

>Terminal:
```console
pip install openai pandas
```

Defina também a variável de ambiente ``OPENAI_API_KEY``. Adicione a linha abaixo ao script de inicialização do seu shell (por exemplo, `.bashrc` ou `.zshrc`) ou execute-a no terminal antes de rodar o código:

>Terminal:
```console
export OPENAI_API_KEY="xx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
Com isso, o ambiente está pronto para acessar a API.

Esta é uma implementação de um chatbot genérico com o GPT-3, pela API da OpenAI:

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


## Passo 4: Preparando o dataset para o fine-tuning

Fine-tuning é o ajuste fino de um modelo para que ele atenda melhor a uma finalidade específica.

Como estudo, vamos fazer o fine-tuning de um chatbot que responde perguntas básicas sobre uma universidade.

Para treinar o modelo, vamos usar o dataset ``intents.json``, disponível no Kaggle: [University Chatbot Dataset, de Niralii Vaghani](https://www.kaggle.com/dsv/5024271 "University Chatbot Dataset - by Niralii vaghani").

### Formatação

Segundo a documentação, o dataset de treinamento mostra ao GPT-3 o que você gostaria que ele respondesse.

Os dados precisam estar no formato [JSONL](https://jsonlines.org/), em que cada linha é um par de prompt e resposta, como neste exemplo:

```JSON
{"prompt": "<prompt text> ->", "completion": " <ideal generated text> \n"}
{"prompt": "<prompt text> ->", "completion": " <ideal generated text> \n"}
{"prompt": "<prompt text> ->", "completion": " <ideal generated text> \n"}
...
```

### Tradução e formatação do dataset
O código a seguir traduz o dataset do inglês para o português usando a API. Ele demora um pouco por causa do limite de 60 requisições por minuto da API:

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

Para não gastar o crédito gratuito da plataforma, traduzi o dataset original por fora e salvei o resultado como `intents_ptBR.json`.

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

O código de reformatação é o mesmo de antes, só sem a tradução pela função ``translateGPT()``:

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

Depois de transformado, o dataset deve ser salvo em um arquivo JSONL, como pede a documentação:

>Python:
```python
#salvando dataset formatado para o arquivo 'dataframe.jsonl'
with open('dataframe.jsonl', 'w', encoding='utf-8') as file:
    file.write(df_output.to_json(orient='records', lines=True, force_ascii=False))
```

### Checando a qualidade do dataframe
A biblioteca OpenAI vem com uma ferramenta que confere a qualidade do dataset antes do upload para o fine-tuning.

O comando ``openai tools fine_tunes.prepare_data -f`` analisa o arquivo JSONL. Para o nosso arquivo, respondemos ``y`` para remover as duplicatas, ``n`` para não dividir o arquivo em conjuntos de treino e validação (o dataset não é grande o bastante para isso) e ``y`` para salvar as mudanças num novo arquivo, ``'dataframe_prepared.jsonl'``. A saída é parecida com esta:

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

Checando o novo dataframe criado:

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


Rodando o comando de novo, vemos a saída de um dataframe formatado corretamente:

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

## Passo 5: Upload do dataset para o fine-tuning

O upload do dataframe tratado pode ser feito pela ferramenta de linha de comando da OpenAI ou pela biblioteca, como no exemplo a seguir. O arquivo enviado recebe um ID, que precisamos guardar para criar o fine-tuning:

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


### Criando o fine-tuning
Depois do upload, treinamos o modelo com a biblioteca OpenAI.

Os parâmetros são o ``training_file``, com o ID do arquivo enviado, e o ``model``, o modelo-base do fine-tuning, que no nosso caso é o ``davinci``.

Guarde o ID do fine-tuning para acompanhar o andamento do treinamento:

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
Visualizando o ID:

>Python:
```python
print(finetune_id)
```

```
ft-iMXzAEiUcFE4nZLDd5kp6tCY
```

Para acompanhar o treinamento, use a ferramenta de linha de comando com o ID do fine-tuning. Execute mais de uma vez até o fim. Quando o treinamento termina, a saída mostra o ID do modelo gerado:

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

Recuperando o ``fine_tuned_model`` do último fine-tuning (confira na etapa anterior que ele terminou):

>Python:
```python
finetune_list = json.loads(json.dumps(openai.FineTune.list()))
finetuned_model_id = finetune_list['data'][-1]['fine_tuned_model']
print(finetuned_model_id)
```
```
    'davinci:ft-personal-2023-04-16-00-36-03'
```


### Testando o modelo treinado

Para testar o modelo, basta abrir um novo chat passando o ``finetuned_model_id`` como engine.

Este foi o resultado do teste:

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


## Conclusão

Consegui criar um chatbot com respostas razoáveis mesmo com um dataset pequeno, de aproximadamente 500 entradas. Com um dataset maior, os resultados devem ser ainda melhores.

### Pontos observados

- A documentação da OpenAI é um pouco falha: mistura instruções da biblioteca Python com requisições HTTP no mesmo lugar, o que dificulta encontrar as referências.
- O treinamento do fine-tuning é barato (este custou US$ 2,01), e usar os modelos prontos ou treinados custa menos ainda.
- O teste gratuito dá um saldo de 18 dólares, o que permite fazer muita coisa legal com esse custo baixo.
- A maior parte do esforço vai para o tratamento do dataset, porque um dataset mal elaborado piora a qualidade das respostas.

### Possibilidades futuras

Neste caso de uso não criei prompts mais elaborados para melhorar as respostas. Dá para fazer isso, seguindo as boas práticas da documentação, e eu recomendo.