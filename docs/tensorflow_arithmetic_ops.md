# Operações Aritméticas no TensorFlow

## Visão Geral

As operações aritméticas básicas (+, -, *, /) foram implementadas para o `FlowTensors` usando os traits `std::ops` do Rust.

## Funcionalidades Implementadas

### Operadores Suportados

- **Adição (`+`)**: Soma elemento a elemento de dois tensores
- **Subtração (`-`)**: Subtração elemento a elemento de dois tensores
- **Multiplicação (`*`)**: Multiplicação elemento a elemento de dois tensores (não é multiplicação de matrizes)
- **Divisão (`/`)**: Divisão elemento a elemento de dois tensores

### Características

- ✅ Operações elemento a elemento (element-wise)
- ✅ Suporte para referências (`&FlowTensors`)
- ✅ Suporte para valores owned (`FlowTensors`)
- ✅ Validação de dimensões
- ✅ Tratamento de divisão por zero (retorna INFINITY)
- ✅ Retorna `Option<FlowTensors>` para tratamento de erros

## Uso

### Exemplo Básico

```rust
use ai_copper::tensor_tensorflow::tensors_flow::FlowTensors;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Criar tensores
    let a = FlowTensors::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .ok_or("Falha ao criar tensor a")?;
    let b = FlowTensors::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])
        .ok_or("Falha ao criar tensor b")?;

    // Adição
    let sum = (&a + &b).ok_or("Falha na adição")?;
    println!("Soma: {:?}", sum.data());
    // Resultado: [6.0, 8.0, 10.0, 12.0]

    // Subtração
    let diff = (&b - &a).ok_or("Falha na subtração")?;
    println!("Subtração: {:?}", diff.data());
    // Resultado: [4.0, 4.0, 4.0, 4.0]

    // Multiplicação (elemento a elemento)
    let prod = (&a * &b).ok_or("Falha na multiplicação")?;
    println!("Multiplicação: {:?}", prod.data());
    // Resultado: [5.0, 12.0, 21.0, 32.0]

    // Divisão
    let quot = (&b / &a).ok_or("Falha na divisão")?;
    println!("Divisão: {:?}", quot.data());
    // Resultado: [5.0, 3.0, 2.333..., 2.0]

    Ok(())
}
```

### Operações Encadeadas

```rust
fn example() -> Result<(), Box<dyn std::error::Error>> {
    let a = FlowTensors::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .ok_or("Falha ao criar tensor a")?;
    let b = FlowTensors::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])
        .ok_or("Falha ao criar tensor b")?;
    let c = FlowTensors::ones(&[2, 2])
        .ok_or("Falha ao criar tensor c")?;

    // (A + B) * C
    if let Some(temp) = &a + &b {
        if let Some(result) = &temp * &c {
            println!("Resultado: {:?}", result.data());
        }
    }
    
    Ok(())
}
```

### Tratamento de Erros

```rust
fn example() -> Result<(), Box<dyn std::error::Error>> {
    let a = FlowTensors::new(&[1.0, 2.0], &[2])
        .ok_or("Falha ao criar tensor a")?;
    let b = FlowTensors::new(&[3.0, 4.0, 5.0], &[3])
        .ok_or("Falha ao criar tensor b")?;

    // Dimensões incompatíveis - retorna None
    match &a + &b {
        Some(result) => println!("Sucesso: {:?}", result.data()),
        None => println!("Erro: Dimensões incompatíveis"),
    }
    
    Ok(())
}
```

## Implementação Técnica

### Lado Rust

Os operadores são implementados usando os traits `std::ops`:

```rust
impl Add for &FlowTensors {
    type Output = Option<FlowTensors>;
    
    fn add(self, other: &FlowTensors) -> Self::Output {
        unsafe {
            let result_ptr = crate::tensor_tensorflow::ffi::TFTensorAdd(self.ptr, other.ptr);
            if result_ptr.is_null() {
                return None;
            }
            Some(FlowTensors {
                ptr: result_ptr,
                dims: self.dims.clone(),
            })
        }
    }
}
```

### Lado C++

As operações são implementadas usando a API C do TensorFlow:

```cpp
EXPORT void* TFTensorAdd(void* tensor_a_ptr, void* tensor_b_ptr) {
    // Validação de ponteiros
    // Verificação de tipos (TF_FLOAT)
    // Verificação de dimensões
    // Alocação de memória para resultado
    // Operação elemento a elemento
    // Criação do tensor resultado
}
```

## Validações Realizadas

1. **Ponteiros válidos**: Verifica se os tensores não são nulos
2. **Tipo de dados**: Garante que ambos os tensores são `TF_FLOAT`
3. **Número de dimensões**: Ambos os tensores devem ter o mesmo número de dimensões
4. **Tamanho das dimensões**: Cada dimensão deve ter o mesmo tamanho em ambos os tensores
5. **Divisão por zero**: Tratada com retorno de `INFINITY`

## Limitações Atuais

- ⚠️ Suporta apenas tipo `f32` (TF_FLOAT)
- ⚠️ Não há broadcasting automático
- ⚠️ Multiplicação é elemento a elemento (não é multiplicação de matrizes)
- ⚠️ Operações in-place não são suportadas

## Executar Exemplo

```bash
cargo run --example tensorflow_arithmetic
```

## Próximos Passos

- [ ] Implementar MatMul (multiplicação de matrizes)
- [ ] Adicionar broadcasting
- [ ] Suportar outros tipos de dados (f64, i32, etc.)
- [ ] Implementar operações in-place
- [ ] Adicionar funções matemáticas (pow, sqrt, exp, log, etc.)
