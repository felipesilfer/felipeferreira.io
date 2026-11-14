# felipeferreira.io

Site pessoal publicado em [felipeferreira.io](https://felipeferreira.io/), feito com [Hugo](https://gohugo.io/) e o tema [Hextra](https://github.com/imfing/hextra).

Criado a partir do [hextra-starter-template](https://github.com/imfing/hextra-starter-template).

## Desenvolvimento local

Pré-requisitos: [Go](https://go.dev/doc/install) e [Hugo extended](https://gohugo.io/installation/).

```shell
hugo mod tidy
hugo server --logLevel debug --disableFastRender -p 1313
```

## Atualizar o tema

```shell
hugo mod get -u
hugo mod tidy
```
