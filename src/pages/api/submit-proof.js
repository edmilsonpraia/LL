import fs from 'fs/promises';
import path from 'path';

export async function POST({ request }) {
  try {
    const formData = await request.formData();
    const email = formData.get('email');
    const name = formData.get('name');
    const proofFile = formData.get('proof');

    // Validações
    if (!email || !name || !proofFile) {
      return new Response(
        JSON.stringify({ error: 'Todos os campos são obrigatórios' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Valida tipo de arquivo
    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf'];
    if (!allowedTypes.includes(proofFile.type)) {
      return new Response(
        JSON.stringify({ error: 'Formato de arquivo não permitido. Use JPG, PNG ou PDF.' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Valida tamanho (5MB)
    if (proofFile.size > 5 * 1024 * 1024) {
      return new Response(
        JSON.stringify({ error: 'Arquivo muito grande. Máximo 5MB.' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Cria pasta de uploads se não existir
    const uploadsDir = path.join(process.cwd(), 'uploads', 'proofs');
    try {
      await fs.mkdir(uploadsDir, { recursive: true });
    } catch (err) {
      console.error('Erro ao criar pasta:', err);
    }

    // Salva arquivo
    const timestamp = Date.now();
    const fileExt = proofFile.name.split('.').pop();
    const fileName = `${timestamp}-${email.replace(/[^a-z0-9]/gi, '_')}.${fileExt}`;
    const filePath = path.join(uploadsDir, fileName);

    const arrayBuffer = await proofFile.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    await fs.writeFile(filePath, buffer);

    // Salva informações no arquivo JSON
    const dataFile = path.join(process.cwd(), 'uploads', 'pending-payments.json');
    let payments = [];

    try {
      const fileContent = await fs.readFile(dataFile, 'utf-8');
      payments = JSON.parse(fileContent);
    } catch (err) {
      // Arquivo não existe ainda, criar novo
    }

    payments.push({
      id: timestamp,
      email,
      name,
      fileName,
      status: 'pending',
      createdAt: new Date().toISOString(),
    });

    await fs.writeFile(dataFile, JSON.stringify(payments, null, 2));

    return new Response(
      JSON.stringify({
        success: true,
        message: 'Comprovante enviado com sucesso!',
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    console.error('Erro ao processar comprovante:', error);
    return new Response(
      JSON.stringify({ error: 'Erro interno do servidor' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
