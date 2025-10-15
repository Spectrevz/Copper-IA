use ai_copper::tensor_tensorflow::tensors_flow::FlowTensors;

fn main() {
    println!("=== Teste de Operações Aritméticas do TensorFlow ===\n");
    
    // Criar dois tensores para teste
    let a = FlowTensors::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("Falha ao criar tensor A");
    let b = FlowTensors::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])
        .expect("Falha ao criar tensor B");
    
    println!("Tensor A:");
    if let Some(data_a) = a.data() {
        println!("  Valores: {:?}", data_a);
        println!("  Dimensões: {:?}", a.dims());
    }
    
    println!("\nTensor B:");
    if let Some(data_b) = b.data() {
        println!("  Valores: {:?}", data_b);
        println!("  Dimensões: {:?}", b.dims());
    }
    
    // Teste de adição
    println!("\n--- ADIÇÃO (A + B) ---");
    if let Some(sum) = &a + &b {
        if let Some(data) = sum.data() {
            println!("Resultado: {:?}", data);
            println!("Esperado:  [6.0, 8.0, 10.0, 12.0]");
        }
    } else {
        println!("Erro ao realizar adição");
    }
    
    // Teste de subtração
    println!("\n--- SUBTRAÇÃO (B - A) ---");
    if let Some(diff) = &b - &a {
        if let Some(data) = diff.data() {
            println!("Resultado: {:?}", data);
            println!("Esperado:  [4.0, 4.0, 4.0, 4.0]");
        }
    } else {
        println!("Erro ao realizar subtração");
    }
    
    // Teste de multiplicação
    println!("\n--- MULTIPLICAÇÃO (A * B) ---");
    if let Some(prod) = &a * &b {
        if let Some(data) = prod.data() {
            println!("Resultado: {:?}", data);
            println!("Esperado:  [5.0, 12.0, 21.0, 32.0]");
        }
    } else {
        println!("Erro ao realizar multiplicação");
    }
    
    // Teste de divisão
    println!("\n--- DIVISÃO (B / A) ---");
    if let Some(quot) = &b / &a {
        if let Some(data) = quot.data() {
            println!("Resultado: {:?}", data);
            println!("Esperado:  [5.0, 3.0, 2.333..., 2.0]");
        }
    } else {
        println!("Erro ao realizar divisão");
    }
    
    // Teste com operações encadeadas
    println!("\n--- OPERAÇÕES ENCADEADAS ---");
    let c = FlowTensors::ones(&[2, 2]).expect("Falha ao criar tensor C");
    
    // (A + B) * C
    if let Some(temp) = &a + &b {
        if let Some(result) = &temp * &c {
            if let Some(data) = result.data() {
                println!("(A + B) * C = {:?}", data);
                println!("Esperado:    [6.0, 8.0, 10.0, 12.0]");
            }
        }
    }
    
    // Teste com valores especiais
    println!("\n--- DIVISÃO POR ZERO ---");
    let zero = FlowTensors::zeros(&[2, 2]).expect("Falha ao criar tensor zero");
    if let Some(inf_result) = &a / &zero {
        if let Some(data) = inf_result.data() {
            println!("A / 0 = {:?}", data);
            println!("(Deve conter valores infinitos)");
        }
    }
    
    println!("\n=== Testes concluídos! ===");
}
