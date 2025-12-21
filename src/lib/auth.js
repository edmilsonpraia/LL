/**
 * Sistema de autenticação baseado em tokens assinados com HMAC
 */

import crypto from 'crypto';

const JWT_SECRET = import.meta.env.JWT_SECRET || 'default-secret-key-change-in-production-2024';

// Função para criar um HMAC signature
function createSignature(data) {
  return crypto
    .createHmac('sha256', JWT_SECRET)
    .update(data)
    .digest('base64url');
}

// Função para criar um token de acesso
export function createAccessToken(email, daysValid = 365) {
  const payload = {
    email,
    timestamp: Date.now(),
    exp: Date.now() + (daysValid * 24 * 60 * 60 * 1000)
  };

  const payloadBase64 = Buffer.from(JSON.stringify(payload)).toString('base64url');
  const signature = createSignature(payloadBase64);

  return `${payloadBase64}.${signature}`;
}

// Função para verificar um token de acesso
export function verifyAccessToken(token) {
  try {
    if (!token || typeof token !== 'string') {
      return null;
    }

    const parts = token.split('.');
    if (parts.length !== 2) {
      return null;
    }

    const [payloadBase64, signature] = parts;

    // Verifica a assinatura
    const expectedSignature = createSignature(payloadBase64);
    if (signature !== expectedSignature) {
      return null; // Assinatura inválida
    }

    // Decodifica o payload
    const payload = JSON.parse(Buffer.from(payloadBase64, 'base64url').toString());

    // Verifica se o token não expirou
    if (payload.exp && payload.exp < Date.now()) {
      return null;
    }

    return payload;
  } catch (error) {
    console.error('Error verifying token:', error);
    return null;
  }
}

// Função para verificar acesso em uma requisição
export function checkAccess(request) {
  // Primeiro tenta pegar o token da URL
  const url = new URL(request.url);
  let accessToken = url.searchParams.get('token');

  // Se não tiver na URL, tenta pegar dos cookies
  if (!accessToken) {
    const cookies = request.headers.get('cookie') || '';
    accessToken = cookies
      .split(';')
      .find(c => c.trim().startsWith('access_token='))
      ?.split('=')[1];
  }

  if (!accessToken) {
    return null;
  }

  return verifyAccessToken(accessToken);
}

// Função para criar cookie de acesso
export function createAccessCookie(token, daysValid = 365) {
  const maxAge = daysValid * 24 * 60 * 60; // em segundos
  return `access_token=${token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${maxAge}`;
}
