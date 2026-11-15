# felipeferreira.io

Site pessoal e blog publicado em [felipeferreira.io](https://felipeferreira.io/), feito com [Hugo](https://gohugo.io/) e o tema [Hextra](https://github.com/imfing/hextra) (módulo Go).

## Desenvolvimento local

Pré-requisitos: [Go](https://go.dev/doc/install) e [Hugo extended](https://gohugo.io/installation/) 0.156+.

```shell
hugo mod tidy
hugo server -p 1313
```

## Estrutura

- `hugo.yaml`: configuração, idiomas (`pt` na raiz, `en` em `/en/`), menus e parâmetros do perfil.
- `content/`: páginas e artigos. A versão em inglês fica ao lado do original, com sufixo `.en.md`.
- `i18n/`: textos dos templates em cada idioma.
- `layouts/`: templates próprios e cópias customizadas do Hextra:
  - `profile.html`: página inicial em modo perfil;
  - `blog/`, `term.html`, `taxonomy.html`: lista de artigos e tags em cartões;
  - `404.html`: página de erro;
  - `_partials/navbar.html` e `_partials/footer.html`: cópias do Hextra v0.12.3 com o seletor PT | EN e o botão de tema;
  - `_partials/components/disqus.html`: comentários pelo Disqus.
- `assets/css/custom.css`: estilos do site.
- `static/`: imagens e favicons.

## Publicação

Cada push na `main` publica o site no GitHub Pages pelo workflow `.github/workflows/pages.yaml`. O domínio fica na Cloudflare.

## Atualizar o tema

```shell
hugo mod get -u
hugo mod tidy
```

Depois de atualizar, compare `layouts/_partials/navbar.html` e `layouts/_partials/footer.html` com as versões novas do Hextra e reaplique as mudanças.
