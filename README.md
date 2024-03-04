
![Trained agent](storage/step-1000000.mp4)

### Requirements básicos e extras

O básico é necessário para executar qualquer arquivo do projeto.
```python
pip install -r requirements.txt           # [básico] Para executar main.py ou algum tests do projeto
pip install -r MARL-requirements.txt      # [extra] Para executar MARL_codebase_main.py 
pip install -r RAY-RLLIB-requirements.txt # [extra] Para executar algum ray_rllib_scripts
```

pytorch para CUDA precisa ser baixado com o comando nesse link:
https://pytorch.org/get-started/locally/

A lista completa de bibliotecas usadas no treinamento estão no arquivo `freeze.txt`

### Como executar o projeto ?

```python
python main.py               # Roda ações de movimento aleatórias com visualização usando pygame
python MARL_codebase_main.py # Treina com dqn. Explicação detalhada dentro do arquivo.
```

Para executar algum arquivo dentro de `tests` é necessário mover esse arquivo para gym_soccer_env.
```python
python actions_test.py         # Testa uma sequênciwa de ações e coloca imagens de cada estado na pasta teste_imgs
python petting_zoo_api_test.py # Executa o teste de API do pedtingzoo
```

### Erros e soluções encontrados

1. Erro ao tentar buildar algum pacote com `pip install -r requirements.txt`

Solução: Baixar o instalador do VS Studio e baixar *Desenvolvimento para desktop com C++*. Isso vai baixar o SDK do windows com o compilador usado para buildar o pacote.

Links úteis:
- https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst
- https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Erro ao tentar `pip install -r MARL-requirements.txt`

Solução: É necessário usar Python versão 9 ou anterior. Python 12 (latest stable, 24/02/2024) não vem com setuptools instalado e, mesmo depois de instalado, tem modificações que geram erros (ex. ModuleNotFoundError: setuptools.extern.six). A versão do Numpy requerida no código MARL, `Numpy==1.19.5`, não suporta Python 11 ou 10.

3. AttributeError: module 'utils' has no attribute 'envs', AttributeError: module 'gym.wrappers' has no attribute 'Monitor'

Solução: `pip install gym==0.15.3`

Links úteis:
- https://stackoverflow.com/questions/71411045/how-to-solve-module-gym-wrappers-has-no-attribute-monitor
