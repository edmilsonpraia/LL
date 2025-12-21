# 💳 Sistema de Pagamento via PIX

Este sistema permite que os usuários comprem o ebook pagando via PIX e enviando comprovante para aprovação manual.

## 🎯 Como Funciona

### Para o Cliente:

1. Acessa a página de pagamento (`/pagamento`)
2. Visualiza QR Code e chave PIX
3. Faz o pagamento via PIX
4. Envia comprovante (foto ou PDF)
5. Aguarda aprovação (até 24h)
6. Recebe email com acesso ao ebook

### Para o Admin:

1. Acessa o painel admin (`/admin`)
2. Visualiza pagamentos pendentes
3. Clica no comprovante para verificar
4. Aprova ou rejeita o pagamento
5. Sistema libera acesso automaticamente

## 📁 Estrutura de Arquivos

```
├── src/
│   ├── pages/
│   │   ├── pagamento.astro          # Página de pagamento PIX
│   │   ├── admin/
│   │   │   └── index.astro          # Painel administrativo
│   │   └── api/
│   │       ├── submit-proof.js      # API para enviar comprovante
│   │       └── manage-payment.js    # API para aprovar/rejeitar
│   └── lib/
│       └── auth.js                  # Sistema de autenticação
└── uploads/                         # Pasta criada automaticamente
    ├── proofs/                      # Comprovantes enviados
    ├── pending-payments.json        # Pagamentos pendentes
    └── approved-users.json          # Usuários aprovados
```

## ⚙️ Configuração

### 1. Configure suas informações PIX no [.env](.env):

```env
PIX_KEY=sua_chave_pix@email.com
PIX_NAME=Seu Nome ou Nome da Empresa
EBOOK_PRICE=9900
SUPPORT_EMAIL=suporte@seusite.com
```

### 2. Gere um QR Code PIX:

Você pode gerar QR Codes PIX usando:

- **Site**: https://www.qrcode-monkey.com/
- **Aplicativo do banco**
- **Ferramenta online**: https://pix.nascent.com.br/tools/pix-qr-code

Substitua o placeholder no código em `src/pages/pagamento.astro` pelo QR Code real.

## 🔐 Segurança

### Proteção de Arquivos

Os comprovantes são salvos em `uploads/proofs/` com nomes únicos:
- Formato: `timestamp-email.extensão`
- Exemplo: `1703010000000-usuario_email_com.jpg`

### Validações Implementadas:

✅ Formatos aceitos: JPG, PNG, PDF
✅ Tamanho máximo: 5MB
✅ Email obrigatório
✅ Nome obrigatório

## 🛠️ Como Usar

### Acessar Painel Admin:

```
http://localhost:4321/admin
```

### Ver Comprovantes:

Os comprovantes ficam em:
```
http://localhost:4321/uploads/proofs/nome-do-arquivo.jpg
```

### Aprovar Pagamento:

1. Entre no painel admin
2. Veja os pagamentos pendentes
3. Clique em "Ver Comprovante"
4. Verifique se o pagamento é válido
5. Clique em "Aprovar"
6. O usuário receberá acesso automaticamente

## 📧 Notificações por Email (Opcional)

Para enviar emails automáticos quando aprovar/rejeitar, adicione em `src/pages/api/manage-payment.js`:

```javascript
// Exemplo usando Nodemailer
import nodemailer from 'nodemailer';

async function sendApprovalEmail(email, accessToken) {
  const transporter = nodemailer.createTransport({
    // Configure seu servidor SMTP
  });

  await transporter.sendMail({
    to: email,
    subject: 'Acesso ao Ebook Liberado!',
    html: `
      <h1>Seu acesso foi aprovado!</h1>
      <p>Clique no link abaixo para acessar o ebook:</p>
      <a href="https://seusite.com/ebook?token=${accessToken}">Acessar Ebook</a>
    `
  });
}
```

## 🎨 Personalização

### Alterar Valor do Ebook:

Edite no [.env](.env):
```env
EBOOK_PRICE=14900  # R$ 149,00
```

### Customizar Página de Pagamento:

Edite `src/pages/pagamento.astro` para:
- Mudar cores
- Adicionar mais informações
- Alterar textos

### Customizar Painel Admin:

Edite `src/pages/admin/index.astro` para:
- Adicionar filtros
- Exportar relatórios
- Adicionar busca

## 📊 Relatórios

### Ver Todos os Pagamentos:

Abra o arquivo `uploads/pending-payments.json`

### Ver Usuários Aprovados:

Abra o arquivo `uploads/approved-users.json`

### Estatísticas:

O painel admin mostra automaticamente:
- Total de pendentes
- Total de aprovados
- Total de rejeitados

## 🔄 Fluxo Completo

```
Cliente acessa /pagamento
    ↓
Faz PIX
    ↓
Envia comprovante
    ↓
Comprovante salvo em uploads/proofs/
    ↓
Dados salvos em pending-payments.json
    ↓
Admin acessa /admin
    ↓
Visualiza e verifica comprovante
    ↓
Aprova pagamento
    ↓
Sistema cria token de acesso
    ↓
Token salvo em approved-users.json
    ↓
Cliente pode acessar /ebook
```

## ⚠️ Importante

1. **Backup**: Faça backup regular da pasta `uploads/`
2. **Git**: Adicione `uploads/` no `.gitignore`
3. **Permissões**: Certifique-se que a pasta tem permissão de escrita
4. **Email de Suporte**: Configure um email real para suporte

## 🆘 Solução de Problemas

### Erro ao enviar comprovante:

- Verifique se a pasta `uploads/` existe
- Verifique permissões de escrita
- Tamanho do arquivo (máx 5MB)

### Painel admin vazio:

- Verifique se existe `uploads/pending-payments.json`
- Verifique se algum comprovante foi enviado

### QR Code não aparece:

- Gere um QR Code real usando sua chave PIX
- Substitua o placeholder em `pagamento.astro`

## 🚀 Melhorias Futuras

- [ ] Integração com API de email
- [ ] Dashboard com gráficos
- [ ] Exportar relatórios em PDF
- [ ] Notificações push
- [ ] Sistema de busca de pagamentos
- [ ] Verificação automática de PIX (API do banco)

---

**Desenvolvido para facilitar a venda de ebooks com pagamento via PIX** 🎉
