#!/usr/bin/env node

/**
 * Script para gerar links de acesso ao ebook
 *
 * Uso:
 *   node generate-access-link.js cliente@email.com
 *   node generate-access-link.js cliente@email.com 30
 *
 * Argumentos:
 *   - email: Email do cliente (obrigatório)
 *   - dias: Número de dias de validade (opcional, padrão: 365 dias)
 */

import crypto from 'crypto';

// Pega o secret do ambiente ou usa o padrão (DEVE ser o mesmo do .env)
const JWT_SECRET = process.env.JWT_SECRET || 'default-secret-key-change-in-production-2024';

// Obtém o domínio do site (pode ser configurado)
const SITE_URL = process.env.SITE_URL || 'http://localhost:4326';

// Função para criar assinatura HMAC
function createSignature(data) {
  return crypto
    .createHmac('sha256', JWT_SECRET)
    .update(data)
    .digest('base64url');
}

// Função para criar token de acesso
function createAccessToken(email, daysValid = 365) {
  const payload = {
    email,
    timestamp: Date.now(),
    exp: Date.now() + (daysValid * 24 * 60 * 60 * 1000)
  };

  const payloadBase64 = Buffer.from(JSON.stringify(payload)).toString('base64url');
  const signature = createSignature(payloadBase64);

  return `${payloadBase64}.${signature}`;
}

// Processa argumentos da linha de comando
const args = process.argv.slice(2);

if (args.length === 0) {
  console.error('\n❌ Erro: Email é obrigatório!\n');
  console.log('Uso:');
  console.log('  node generate-access-link.js <email> [dias]\n');
  console.log('Exemplos:');
  console.log('  node generate-access-link.js cliente@email.com');
  console.log('  node generate-access-link.js cliente@email.com 30');
  console.log('  node generate-access-link.js cliente@email.com 365\n');
  process.exit(1);
}

const email = args[0];
const daysValid = args[1] ? parseInt(args[1]) : 365;

// Valida o email
const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
if (!emailRegex.test(email)) {
  console.error('\n❌ Erro: Email inválido!\n');
  process.exit(1);
}

// Valida os dias
if (isNaN(daysValid) || daysValid <= 0) {
  console.error('\n❌ Erro: Número de dias inválido!\n');
  process.exit(1);
}

// Gera o token
const token = createAccessToken(email, daysValid);

// Gera o link completo
const accessLink = `${SITE_URL}/ebook?token=${token}`;

// Calcula a data de expiração
const expirationDate = new Date(Date.now() + (daysValid * 24 * 60 * 60 * 1000));

// Exibe o resultado
console.log('\n' + '='.repeat(80));
console.log('🎉 LINK DE ACESSO GERADO COM SUCESSO!');
console.log('='.repeat(80));
console.log('\n📧 Email do Cliente:', email);
console.log('⏱️  Validade:', daysValid, 'dias');
console.log('📅 Expira em:', expirationDate.toLocaleDateString('pt-BR'), 'às', expirationDate.toLocaleTimeString('pt-BR'));
console.log('\n🔗 Link de Acesso:');
console.log('─'.repeat(80));
console.log(accessLink);
console.log('─'.repeat(80));
console.log('\n📋 Instruções para o Cliente:');
console.log('1. Clique no link acima');
console.log('2. O acesso será liberado automaticamente');
console.log('3. Um cookie será salvo no navegador para acesso futuro');
console.log('4. Para acessar novamente, basta ir em: ' + SITE_URL + '/ebook');
console.log('\n💡 Dica: Você pode enviar este link por WhatsApp ou email\n');
console.log('='.repeat(80) + '\n');
