/**
 * Configuração do Stripe para processamento de pagamentos
 */

import Stripe from 'stripe';

const stripeSecretKey = import.meta.env.STRIPE_SECRET_KEY;

if (!stripeSecretKey) {
  console.warn('STRIPE_SECRET_KEY não configurada. Configure no arquivo .env');
}

export const stripe = stripeSecretKey
  ? new Stripe(stripeSecretKey, {
      apiVersion: '2023-10-16',
    })
  : null;

export const EBOOK_PRICE = parseInt(import.meta.env.EBOOK_PRICE || '9900');

/**
 * Cria uma sessão de checkout do Stripe
 */
export async function createCheckoutSession(origin) {
  if (!stripe) {
    throw new Error('Stripe não configurado');
  }

  const session = await stripe.checkout.sessions.create({
    payment_method_types: ['card'],
    line_items: [
      {
        price_data: {
          currency: 'brl',
          product_data: {
            name: 'PINNs em Petrofísica',
            description: 'Integração entre Deep Learning e Equações de Rocha',
            images: [`${origin}/ebook-cover.png`],
          },
          unit_amount: EBOOK_PRICE,
        },
        quantity: 1,
      },
    ],
    mode: 'payment',
    success_url: `${origin}/success?session_id={CHECKOUT_SESSION_ID}`,
    cancel_url: `${origin}/?canceled=true`,
    metadata: {
      product: 'pinns-petrofisica-ebook'
    }
  });

  return session;
}

/**
 * Verifica o status de uma sessão de checkout
 */
export async function verifyCheckoutSession(sessionId) {
  if (!stripe) {
    throw new Error('Stripe não configurado');
  }

  const session = await stripe.checkout.sessions.retrieve(sessionId);
  return session;
}
