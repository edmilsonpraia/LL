import { createAccessToken } from '../../lib/auth.js';
import crypto from 'crypto';

// Função para criar hash do token de sessão (deve ser igual ao admin.astro)
function createSessionToken(password) {
  return crypto.createHash('sha256').update(password + 'salt-admin-2024').digest('hex');
}

export async function POST({ request }) {
  try {
    // Verifica autenticação admin
    const cookies = request.headers.get('cookie') || '';
    const adminPassword = import.meta.env.ADMIN_PASSWORD || 'admin123';
    const sessionToken = cookies
      .split(';')
      .find(c => c.trim().startsWith('admin_session='))
      ?.split('=')[1];

    const expectedToken = createSessionToken(adminPassword);
    const isAuthenticated = sessionToken === expectedToken;

    if (!isAuthenticated) {
      return new Response(JSON.stringify({ error: 'Não autorizado' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Pega os dados do body
    const { email, days, ebook } = await request.json();

    // Valida email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email || !emailRegex.test(email)) {
      return new Response(JSON.stringify({ error: 'Email inválido' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Valida ebook
    const validEbooks = ['pinns-petrofisica', 'integracao-de-metodos', 'seismic-ml'];
    if (!ebook || !validEbooks.includes(ebook)) {
      return new Response(JSON.stringify({ error: 'Ebook inválido ou não especificado' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Valida dias
    const daysValid = parseInt(days) || 365;
    if (daysValid <= 0 || daysValid > 36500) {
      return new Response(JSON.stringify({ error: 'Número de dias inválido' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Gera o token
    const token = createAccessToken(email, daysValid);

    // Monta o link com o slug do ebook
    const siteUrl = import.meta.env.SITE_URL || 'http://localhost:4326';
    const link = `${siteUrl}/ebook/${ebook}?token=${token}`;

    // Calcula data de expiração
    const expiryDate = new Date(Date.now() + (daysValid * 24 * 60 * 60 * 1000));

    return new Response(
      JSON.stringify({
        success: true,
        link,
        email,
        days: daysValid,
        expiryDate: expiryDate.toISOString(),
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  } catch (error) {
    console.error('Error generating link:', error);
    return new Response(
      JSON.stringify({ error: 'Erro interno ao gerar link' }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}
