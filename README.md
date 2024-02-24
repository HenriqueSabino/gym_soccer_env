
### Como rodar um algoritmo do MARL codebase ?

```
python MARL_codebase_main.py
```

### Erros e soluções encontrados

1. Erro ao tentar buildar algum pacote com `pip install -r requirements.txt`

R: Baixar o instalador do VS Studio e baixar *Desenvolvimento para desktop com C++* (~7GiB)

Links úteis:
- https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst
- https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Erro ao tentar `pip install -r MARL-requirements.txt`

R: É necessário usar Python versão 9 ou anterior. Python 12 (latest stable, 24/02/2024) não vem com setuptools instalado e, mesmo depois de instalado, tem modificações que geram erros (ex. ModuleNotFoundError: setuptools.extern.six). A versão do Numpy requerida no código MARL, `Numpy==1.19.5`, não suporta Python 11 ou 10.

3. AttributeError: module 'utils' has no attribute 'envs', AttributeError: module 'gym.wrappers' has no attribute 'Monitor'

R: `pip install gym==0.15.3`

Links úteis:
- https://stackoverflow.com/questions/71411045/how-to-solve-module-gym-wrappers-has-no-attribute-monitor
