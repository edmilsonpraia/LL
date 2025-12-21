# 🎛️ Guia do Painel Administrativo

## 🚀 Acesso ao Painel

**URL Local:** http://localhost:4326/admin
**URL Produção:** https://seu-site.vercel.app/admin

### Primeira Vez

1. Acesse `/admin`
2. Use a senha padrão: `admin123`
3. **IMPORTANTE:** Mude a senha em produção!

### Mudar a Senha

Edite o arquivo `.env`:

```env
ADMIN_PASSWORD=sua_senha_super_secreta_aqui
```

⚠️ **NUNCA** compartilhe esta senha ou faça commit do arquivo `.env`!

## 📋 Como Usar

### 1. Fazer Login

1. Acesse http://localhost:4326/admin
2. Digite a senha do admin
3. Clique em "Entrar"

### 2. Gerar Link de Acesso

Quando um cliente pagar:

1. **Preencha o Email**: Digite o email que o cliente usou no formulário
2. **Escolha a Validade**:
   - 30 dias
   - 90 dias
   - 180 dias
   - 1 ano (padrão)
   - 2 anos
   - Vitalício (10 anos)
3. **Clique em "Gerar Link de Acesso"**

### 3. Enviar Link ao Cliente

Você tem 3 opções:

#### Opção A: Copiar e Colar
1. Clique em "Copiar"
2. Cole o link no WhatsApp/Email do cliente

#### Opção B: WhatsApp Direto
1. Clique em "📱 Enviar via WhatsApp"
2. O WhatsApp abre com mensagem pronta
3. Selecione o contato do cliente
4. Envie

#### Opção C: Email Manual
Copie e envie por email com este template:

```
Assunto: Acesso Liberado - Ebook PINNs em Petrofísica

Olá [Nome],

Seu pagamento foi confirmado com sucesso! 🎉

Clique no link abaixo para acessar o ebook:
[COLE O LINK AQUI]

O link é pessoal e intransferível. Após o primeiro acesso,
você poderá retornar sempre que quiser.

Aproveite seus estudos!

Atenciosamente,
Edmilson Delfim Praia
```

## 🔄 Fluxo Completo

```
1. Cliente paga → Envia comprovante no WhatsApp
                ↓
2. Você verifica o pagamento
                ↓
3. Abre /admin → Gera link com email do cliente
                ↓
4. Copia o link ou usa botão WhatsApp
                ↓
5. Envia para o cliente
                ↓
6. Cliente clica → Acesso liberado automaticamente!
```

## 🔐 Segurança

### O que o Link Contém?

- Email do cliente (criptografado)
- Data de expiração
- Assinatura HMAC (impossível falsificar)

### Como Funciona a Segurança?

1. **Token Assinado**: Usa HMAC SHA-256 com chave secreta
2. **Impossível Falsificar**: Sem a chave secreta (`JWT_SECRET`), ninguém pode criar tokens válidos
3. **Expira Automaticamente**: Após o período definido, o link para de funcionar
4. **Cookie Seguro**: Após primeiro acesso, cookie HttpOnly é salvo
5. **Um email = Um acesso**: Cada link é único para aquele email

### Revogar Todos os Acessos

Se precisar invalidar TODOS os links já gerados:

1. Mude o `JWT_SECRET` no `.env`
2. Todos os links antigos ficarão inválidos
3. Gere novos links para clientes autorizados

## 📱 Deploy na Vercel

### Passo a Passo

1. **Conecte o Repositório**
   - Faça push do código para GitHub
   - Conecte no Vercel

2. **Configure as Variáveis de Ambiente**

   No painel da Vercel, adicione:

   ```
   JWT_SECRET=sua-chave-super-secreta-2024
   ADMIN_PASSWORD=sua-senha-admin-segura
   SITE_URL=https://seu-site.vercel.app
   EBOOK_PRICE=9999
   ```

3. **Deploy**
   - Vercel faz deploy automático
   - Acesse: `https://seu-site.vercel.app/admin`

### Importante na Vercel

✅ **SEMPRE** use variáveis de ambiente para senhas
✅ **NUNCA** faça commit do arquivo `.env`
✅ Use `SITE_URL` da produção nas variáveis de ambiente

## ⚙️ Variáveis de Ambiente

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `ADMIN_PASSWORD` | Senha do painel admin | `minha_senha_123` |
| `JWT_SECRET` | Chave para assinar tokens | `chave-secreta-2024` |
| `SITE_URL` | URL do site | `https://site.vercel.app` |
| `EBOOK_PRICE` | Preço do ebook | `9999` |

## 🐛 Solução de Problemas

### "Não autorizado" ao tentar gerar link

**Causa:** Senha incorreta ou sessão expirada
**Solução:** Faça logout e login novamente

### Link gerado não funciona

**Causa 1:** `SITE_URL` incorreto
**Solução:** Verifique se `SITE_URL` aponta para o domínio correto

**Causa 2:** `JWT_SECRET` foi mudado
**Solução:** Gere um novo link com o novo secret

### Cliente não consegue acessar

**Causa 1:** Link expirado
**Solução:** Gere um novo link

**Causa 2:** Link incompleto (cortado)
**Solução:** Envie novamente, certifique-se que está completo

**Causa 3:** Cookie bloqueado
**Solução:** Cliente deve permitir cookies no navegador

## 📊 Estatísticas

Atualmente, o painel mostra placeholders. Para estatísticas reais:

### Opção 1: Google Analytics
Adicione o código do GA no `<head>` do Layout

### Opção 2: Log Manual
Modifique `src/pages/ebook.astro` para salvar acessos

### Opção 3: Serviço de Analytics
Use: Plausible, Fathom, ou similar

## 💡 Dicas Profissionais

1. **Sempre teste os links** antes de enviar ao cliente
2. **Mantenha registro** dos emails e links gerados (copie e cole num doc)
3. **Use WhatsApp** para envio rápido e confirmação de leitura
4. **Validade recomendada**: 365 dias (1 ano) para clientes normais
5. **Backup da chave**: Guarde `JWT_SECRET` em local seguro

## 🔄 Atualizar Senha Admin

### Desenvolvimento (Local)

1. Edite `.env`:
   ```env
   ADMIN_PASSWORD=nova_senha_aqui
   ```
2. Reinicie o servidor
3. Faça logout e login novamente

### Produção (Vercel)

1. Vá em Settings → Environment Variables
2. Edite `ADMIN_PASSWORD`
3. Redeploy a aplicação
4. Use a nova senha

## 📞 Suporte

**Email:** seen85739@gmail.com
**WhatsApp:** +7 996 100 74 08

---

**Versão:** 1.0.0
**Última atualização:** Dezembro 2024
