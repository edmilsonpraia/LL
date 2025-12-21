import fs from 'fs/promises';
import path from 'path';
import { createAccessToken } from '../../lib/auth.js';

export async function POST({ request }) {
  try {
    const { id, action } = await request.json();

    if (!id || !action) {
      return new Response(
        JSON.stringify({ error: 'ID e ação são obrigatórios' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    if (!['approve', 'reject'].includes(action)) {
      return new Response(
        JSON.stringify({ error: 'Ação inválida' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Lê o arquivo de pagamentos
    const dataFile = path.join(process.cwd(), 'uploads', 'pending-payments.json');
    const fileContent = await fs.readFile(dataFile, 'utf-8');
    const payments = JSON.parse(fileContent);

    // Encontra o pagamento
    const paymentIndex = payments.findIndex((p) => p.id === id);
    if (paymentIndex === -1) {
      return new Response(
        JSON.stringify({ error: 'Pagamento não encontrado' }),
        { status: 404, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const payment = payments[paymentIndex];

    // Atualiza o status
    if (action === 'approve') {
      payment.status = 'approved';
      payment.approvedAt = new Date().toISOString();

      // Cria token de acesso para o usuário
      const accessToken = createAccessToken(payment.email, payment.id);

      // Salva o token de acesso
      const approvedFile = path.join(process.cwd(), 'uploads', 'approved-users.json');
      let approvedUsers = [];

      try {
        const content = await fs.readFile(approvedFile, 'utf-8');
        approvedUsers = JSON.parse(content);
      } catch (err) {
        // Arquivo não existe
      }

      approvedUsers.push({
        email: payment.email,
        name: payment.name,
        accessToken,
        approvedAt: payment.approvedAt,
      });

      await fs.writeFile(approvedFile, JSON.stringify(approvedUsers, null, 2));

      // Aqui você pode enviar um email para o usuário com o link de acesso
      // Exemplo: sendEmail(payment.email, accessToken);
    } else {
      payment.status = 'rejected';
      payment.rejectedAt = new Date().toISOString();

      // Aqui você pode enviar um email informando a rejeição
    }

    // Salva as alterações
    payments[paymentIndex] = payment;
    await fs.writeFile(dataFile, JSON.stringify(payments, null, 2));

    return new Response(
      JSON.stringify({
        success: true,
        message: action === 'approve' ? 'Pagamento aprovado' : 'Pagamento rejeitado',
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    console.error('Erro ao gerenciar pagamento:', error);
    return new Response(
      JSON.stringify({ error: 'Erro interno do servidor' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
