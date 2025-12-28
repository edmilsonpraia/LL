# Estrutura de Imagens dos Ebooks

Esta pasta contém as imagens organizadas por ebook.

## 📁 Organização

```
/public/images/
├── pinns-petrofisica/      # Imagens do ebook PINNs em Petrofísica
│   ├── cover.png           # Capa do ebook
│   ├── Picture1.png        # Imagens do conteúdo
│   ├── Picture2.png
│   └── ...
├── novo-ebook/             # Imagens de um novo ebook (exemplo)
│   ├── cover.png
│   ├── imagem1.png
│   └── ...
└── QR.jpeg                 # QR Code do pagamento (global)
```

## 🎯 Como Adicionar um Novo Ebook

### 1. Criar Pasta de Imagens

```bash
mkdir public/images/nome-do-ebook
```

### 2. Adicionar Imagens

- Coloque a capa como `cover.png`
- Coloque as imagens do conteúdo (podem ter qualquer nome)

### 3. Criar Arquivo MDX

- Copie o template: `src/content/ebook/TEMPLATE-novo-ebook.mdx`
- Renomeie para: `src/content/ebook/nome-do-ebook.mdx`
- Edite o frontmatter e conteúdo

### 4. Referenciar Imagens no MDX

```markdown
![Descrição da imagem](/images/nome-do-ebook/imagem1.png)
```

### 5. Atualizar Referências na Página de Pagamento (se necessário)

Se quiser usar este ebook na página de pagamento:

```astro
<img src="/images/nome-do-ebook/cover.png" alt="Capa" />
```

## 🚀 Acessar o Ebook

O ebook estará disponível em:
```
https://seusite.com/ebook/nome-do-ebook
```

## 📝 Notas

- **Imagens globais** (como QR.jpeg para pagamento) ficam diretamente em `/images/`
- **Imagens específicas de ebook** ficam em `/images/nome-do-ebook/`
- Sempre use caminhos absolutos começando com `/images/...`
- Formatos suportados: PNG, JPG, JPEG, GIF, WebP, SVG

## Dicas

- Use nomes descritivos para suas imagens
- Mantenha os nomes em minúsculas e sem espaços (use hífen ou underscore)
- Otimize suas imagens para web antes de adicionar aqui
