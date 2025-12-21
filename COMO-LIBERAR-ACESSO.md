# Como Liberar Acesso ao Ebook

Este guia explica como liberar acesso ao ebook para seus clientes após receberem o pagamento via WhatsApp.

## 🔄 Fluxo de Pagamento

1. **Cliente faz o pagamento** via transferência bancária BAI
2. **Cliente preenche o formulário** na página de pagamento
3. **WhatsApp abre automaticamente** com uma mensagem contendo:
   - Email do cliente
   - Nome do cliente
   - Valor pago (9999 Kz)
4. **Cliente envia o comprovante** pelo WhatsApp
5. **Você recebe a notificação** no WhatsApp: +7 996 100 74 08
6. **Você verifica o pagamento** e gera o link de acesso
7. **Você envia o link** para o cliente via WhatsApp ou email

## 🔗 Como Gerar Link de Acesso

### Método Simples (Linha de Comando)

Abra o terminal na pasta do projeto e execute:

```bash
node generate-access-link.js cliente@email.com
```

Ou especifique o número de dias de validade:

```bash
node generate-access-link.js cliente@email.com 365
```

### Exemplos:

```bash
# Acesso por 1 ano (365 dias)
node generate-access-link.js joao@email.com 365

# Acesso por 30 dias
node generate-access-link.js maria@email.com 30

# Acesso vitalício (10 anos)
node generate-access-link.js carlos@email.com 3650
```

### O que o script faz:

1. ✅ Valida o email
2. ✅ Cria um token criptografado e assinado (HMAC SHA-256)
3. ✅ Gera um link único e seguro
4. ✅ Exibe o link pronto para enviar ao cliente

### Exemplo de saída:

```
================================================================================
🎉 LINK DE ACESSO GERADO COM SUCESSO!
================================================================================

📧 Email do Cliente: cliente@email.com
⏱️  Validade: 365 dias
📅 Expira em: 21/12/2025 às 10:30:00

🔗 Link de Acesso:
────────────────────────────────────────────────────────────────────────────────
http://localhost:4326/ebook?token=eyJlbWFpbCI6ImNsaWVudGVAZW1haWwuY29tIiwid...
────────────────────────────────────────────────────────────────────────────────

📋 Instruções para o Cliente:
1. Clique no link acima
2. O acesso será liberado automaticamente
3. Um cookie será salvo no navegador para acesso futuro
4. Para acessar novamente, basta ir em: http://localhost:4326/ebook
```

## 📱 Como Enviar o Link para o Cliente

### Via WhatsApp (Recomendado):

```
Olá [Nome]! 👋

Seu pagamento foi confirmado! 🎉

Aqui está seu link de acesso ao ebook "PINNs em Petrofísica":

[COLE O LINK AQUI]

✅ Basta clicar no link para ter acesso imediato
✅ O acesso é vitalício
✅ Você pode acessar de qualquer dispositivo

Aproveite seus estudos! 📚
```

### Via Email:

**Assunto:** Acesso Liberado - Ebook PINNs em Petrofísica

**Corpo:**
```
Olá [Nome],

Seu pagamento foi confirmado com sucesso!

Clique no link abaixo para acessar o ebook:
[COLE O LINK AQUI]

O link é pessoal e intransferível. Após o primeiro acesso, você poderá
retornar ao ebook sempre que quiser através de: http://localhost:4326/ebook

Validade: 365 dias a partir de hoje

Aproveite seus estudos!

Atenciosamente,
Edmilson Delfim Praia
```

## 🔒 Segurança

- ✅ **Tokens assinados com HMAC SHA-256** - impossível falsificar
- ✅ **Expira automaticamente** após o período definido
- ✅ **Um token por email** - não pode ser compartilhado
- ✅ **Cookie HttpOnly** - protegido contra roubo via JavaScript
- ✅ **Sem banco de dados** - tudo funciona com criptografia

## ⚙️ Configuração (Opcional)

Para produção, é recomendado definir um secret key único no arquivo `.env`:

```env
JWT_SECRET=sua-chave-secreta-super-segura-aqui
SITE_URL=https://seusite.com
```

Se não definir, o sistema usa uma chave padrão (funciona, mas menos seguro).

## 🆘 Solução de Problemas

### Cliente diz que o link não funciona:

1. Verifique se o link está completo (não foi cortado)
2. Verifique se o token não expirou
3. Gere um novo link e envie novamente

### Cliente perdeu o acesso:

1. Gere um novo link com o mesmo email
2. O novo link substituirá o anterior

### Como revogar acesso:

1. Mude o `JWT_SECRET` no arquivo `.env`
2. Todos os links anteriores ficarão inválidos
3. Gere novos links para clientes autorizados

## 📊 Estatísticas

Atualmente não há sistema de estatísticas. Para adicionar:
- Considere usar Google Analytics no ebook
- Ou adicione logging no arquivo `src/pages/ebook.astro`

## 🎯 Resumo Rápido

1. Cliente paga → Recebe notificação no WhatsApp
2. Execute: `node generate-access-link.js email@cliente.com`
3. Copie o link gerado
4. Envie para o cliente via WhatsApp ou email
5. Pronto! Cliente tem acesso imediato

---

**Dúvidas?** Entre em contato: seen85739@gmail.com ou WhatsApp: +7 996 100 74 08
