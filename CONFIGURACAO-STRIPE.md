# 🔧 Como Configurar o Stripe

Este guia irá te ajudar a configurar o Stripe para processar pagamentos no site do ebook.

## 📋 Pré-requisitos

- Conta no Stripe (gratuita)
- Acesso ao arquivo `.env` do projeto

## 🚀 Passo a Passo

### 1. Criar conta no Stripe

1. Acesse [stripe.com](https://stripe.com)
2. Clique em "Sign up"
3. Preencha seus dados e crie a conta

### 2. Obter as chaves de API

1. Faça login no [Dashboard do Stripe](https://dashboard.stripe.com)
2. No menu lateral, clique em **"Developers"** (Desenvolvedores)
3. Clique em **"API keys"** (Chaves de API)
4. Você verá duas chaves:
   - **Publishable key** (pk_test_...)
   - **Secret key** (sk_test_...) - clique em "Reveal test key" para ver

### 3. Configurar o arquivo .env

1. Abra o arquivo `.env` na raiz do projeto
2. Cole suas chaves do Stripe:

```env
# Chaves do Stripe
STRIPE_SECRET_KEY=sk_test_sua_chave_secreta_aqui
STRIPE_PUBLISHABLE_KEY=pk_test_sua_chave_publica_aqui

# Preço do Ebook em centavos (9900 = R$ 99,00)
EBOOK_PRICE=9900

# Secret para JWT
JWT_SECRET=supersecretjwtkey12345changeinproduction
```

### 4. Reiniciar o servidor

Depois de configurar, reinicie o servidor:

```bash
# Pressione Ctrl+C para parar o servidor
# Depois execute novamente:
npm run dev
```

## ✅ Verificar se funcionou

1. Acesse `http://localhost:4321/checkout`
2. Se o aviso amarelo desapareceu, está configurado corretamente!
3. Teste um pagamento com cartão de teste:
   - Número: `4242 4242 4242 4242`
   - Data: Qualquer data futura (ex: 12/34)
   - CVV: Qualquer 3 dígitos (ex: 123)
   - CEP: Qualquer código (ex: 12345)

## 💳 Cartões de Teste

O Stripe fornece vários cartões de teste:

- **Sucesso**: `4242 4242 4242 4242`
- **Requer autenticação**: `4000 0025 0000 3155`
- **Cartão recusado**: `4000 0000 0000 9995`

[Ver todos os cartões de teste](https://stripe.com/docs/testing#cards)

## 🌐 Modo Produção

Para usar em produção:

1. No Dashboard do Stripe, clique no toggle "Test mode" para desativá-lo
2. Obtenha as chaves de produção (começam com `pk_live_` e `sk_live_`)
3. Atualize o arquivo `.env` com as chaves de produção
4. **IMPORTANTE**: Nunca commite o arquivo `.env` no Git!

## ❓ Problemas Comuns

### Erro: "Invalid API Key"
- Verifique se copiou a chave completa
- Certifique-se de que não há espaços extras
- Chaves de teste começam com `sk_test_` e `pk_test_`

### Aviso ainda aparece
- Reinicie o servidor (`Ctrl+C` e `npm run dev`)
- Verifique se salvou o arquivo `.env`
- Confirme que está usando `sk_test_` no início da chave secreta

### Pagamento não processa
- Use os cartões de teste fornecidos pelo Stripe
- Verifique se está em modo de teste
- Veja o console do navegador para erros

## 📚 Recursos

- [Documentação do Stripe](https://stripe.com/docs)
- [Cartões de teste](https://stripe.com/docs/testing)
- [Dashboard do Stripe](https://dashboard.stripe.com)

---

**Dica**: Mantenha suas chaves em segredo e nunca as compartilhe publicamente!
