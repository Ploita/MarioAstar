# Adaptado de Fabrício Olivetti de França Mario A* https://folivetti.github.io/courses/IA/ 
# todo adicionar as bibliotecas e criar o enviroment que faz isso funcionar
# todo consertar os #type: ignore

# ? porque tá dando tudo pula se ele não tá pulando na maior parte do tempo?
from __future__ import annotations
import sys
import os
import pickle
import retro # type: ignore
from retro import RetroEnv # type: ignore
from rominfo import *
from utils import *

sys.setrecursionlimit(10000)

# quais movimentos estarão disponíveis
moves = {'direita':128, 'corre':130, 'pula':131, 'spin':386, 'esquerda':64}
moves_int = ['direita', 'corre', 'pula', 'spin', 'esquerda']
# raio de visão (quadriculado raio x raio em torno do Mario)
raio = 6

# Classe da árvore de jogos para o Super Mario World
class Tree:
    """
        Classe da árvore de jogos para o Super Mario World
    """

    def __init__(self, estado: str, action: str, filhos: list[Tree] | None = None, pai: Tree | None = None, g: int = 0,
                  x: int = 0, terminal: bool = False):
        """Inicialização do nó/árvore

        Parameters
        ----------
        estado : str
            Representação N x N da tela do jogo, onde N é o raio de visão
        action : str
            Ação que leva a este estado
        filhos : list[Tree] | None, optional
            Lista de nós filhos deste nó, by default None
        pai : Tree | None, optional
            Nó pai deste nó, by default None
        g : int, optional
            Valor de custo do Nó inicial até o Nó atual, pode ser lido como profundidade da árvore, by default 0
        x : int
            Posição no eixo X do Mario
        terminal : bool, optional
            Indicador se o Nó atual é terminal, by default False
        """

        self.estado = estado
        self.filhos = filhos
        self.g = g
        self.h = self.heuristica(estado, x)
        self.eh_terminal = terminal
        self.eh_obj      = self.checaObj(x)
        self.pai = pai
        self.action = action

    def __str__(self):
        return f"Tree(data={self.g})"
  
    def melhor_filho(self) -> Tree | None:
        """Encontra o melhor filho do nó

        Parameters
        ----------
        tree : Tree
            Nó atual da árvore

        Returns
        -------
        Tree | None
            Melhor nó filho ou None caso não haja nós filhos válidos
        """
        #Se o nó é terminal, retorna None
        if self.eh_terminal:
            return None
        
        # Se o nó não tem filhos, retorna ele mesmo
        if self.filhos is None:
            return self

        # Para cada filho de tree, aplica melhor_filho e filtra aqueles que resultarem em None
        lista_filhos: list[Tree] = []
        for filho in self.filhos:
            no_filho = filho.melhor_filho() 
            if no_filho is not None:
                lista_filhos.append(no_filho) 

        #Se todos os filhos resultarem em terminal, marca o nó como terminal e retorna None
        if not lista_filhos:
            self.eh_terminal = True
            return None

        # Caso contrário, retorna o nó com o menor f
        lista_f = [[filho.g + filho.h, filho] for filho in lista_filhos]
        
        melhor_f =  min(lista_f, key = lambda x: x[0])[1] # type: ignore
        # ? Tem um bug pra veificação de tipo aqui em cima que não soube ajeitar
        
        return melhor_f # type: ignore

    def heuristica(self, estado: str, x: int) -> int:
        """Estima a quantidade de passos mínimos estimados para chegar ao final da fase

        Parameters
        ----------
        estado : str
            Representação N x N da tela do jogo, onde N é o raio de visão
        x : int
            Posição do Mario no eixo X

        Returns
        -------
        int
            Estimativa de passos para chegar ao final da fase
        """
        #//    return (4800 - x)/8
        # todo entender e explicar isso aqui seria uma boa
        estNum = np.reshape(list(map(int, estado.split(','))), (2*raio+1,2*raio+1)) #type: ignore
        dist = np.abs(estNum[:raio+1,raio+2:raio+7]).sum() #type: ignore
        # ? Tem um bug pra veificação de tipo aqui em cima que não soube ajeitar
        return ((4800 - x)/8) + 0.3*dist
    
    def checaObj(self, x: int) -> bool:
        """Verifica se chegou ao final da fase

        Parameters
        ----------
        estado : str
            Representação N x N da tela do jogo, onde N é o raio de visão
        x : int
            Posição do Mario no eixo x

        Returns
        -------
        bool
            Flag se atingiu o objetivo
        """
        return x>4800

    def folha(self) -> bool:
        """ Verifica se tree é um nó folha

        Parameters
        ----------
        tree : Tree
            Nó atual da árvore

        Returns
        -------
        bool
            Flag se é folha
        """
        # Um nó folha é aquele que não tem filhos.
        if self.filhos is None:
            return True
        return False
    def rota(self) -> list[str]:
        node_temp = self.melhor_filho()
        acoes = []
        while node_temp.pai is not None: # type: ignore
            neto = node_temp
            node_temp = node_temp.pai # type: ignore
            acoes.append(neto.action)
            acoes.reverse()
        return acoes


def emula(acoes: list[str], env: RetroEnv, mostrar: bool) -> tuple[str, int, bool]:
    """Joga uma partida usando uma # sequência de ações

    Parameters
    ----------
    acoes : list[str]
        Lista de ações
    env : RetroEnv
        Ambiente do retro Gym
    mostrar : bool
        Flag de visualização

    Returns
    -------
    tuple[str, int, bool]
        Estado atual, posição do Mario no eixo X e flag de finalização do jogo
    """

    env.reset()
    while len(acoes)>0 and (not env.data.is_done()): #type: ignore # ? type error aqui tb
        a = acoes.pop(0)
        estado, xn, y = getState(getRam(env), raio) #type: ignore #? type error
        performAction(a, env)
        #time.sleep(0.023) #ajuda na visualização
        if mostrar:
            env.render()
    over = False
    estado, x, y = getState(getRam(env), raio) #type: ignore #? type erro
    if env.data.is_done() or y > 400: #type: ignore #? type erro
        over = True
    return estado, x, over # type: ignore
# ! o erro de tipo vem das funções em rominfo que não possuem declaração de tipo de retorno
    
# Expande a árvore utilizando a heurística
def expande(tree: Tree, env: RetroEnv, mostrar: bool) -> tuple[Tree, bool]:
    """Expande a árvore utilizando a heurística 

    Parameters
    ----------
    tree : Tree
        Nó raiz
    env : RetroEnv
        Ambiente do retro Gym
    mostrar : bool
        Flag de visualização

    Returns
    -------
    tuple[Tree, bool]
        Nó raiz e flag se atingiu o objetivo
    """
    
    acoes = []
    # Se o nó já for um nó folha não tem ações a serem feitas 
    if tree.folha():
        node_temp  = tree
        filho = tree
    else:
        # Busca pelo melhor nó folha
        filho = tree.melhor_filho()
        node_temp = filho
        # Faz a rota do melhor filho até a raiz
        while node_temp.pai is not None: # type: ignore
            neto = node_temp
            node_temp = node_temp.pai # type: ignore
            acoes.append(neto.action)

    acoes.reverse()
    print('ACOES:  (  ', filho.g, ' ): ',  acoes[200:]) # type: ignore
        
    obj = False

    # Gera cada um dos filhos e verifica se atingiu objetivo
    # todo descobrir o que exatamente tá rolando e atualizar essa parte
    filho.filhos = [] # type: ignore
    maxX         = 0
    for k, v in moves.items():
        estado, x, over = emula([moves[acao] for acao in acoes] + [v], env, mostrar) # type: ignore
        maxX            = max(x, maxX)
        if obj:
            obj = True
        filho.filhos.append(Tree(estado = estado, action = k, g = filho.g + 1, pai= filho, x = x, terminal= over))
        
    print('FALTA: ', filho.heuristica(estado, maxX)) # type: ignore
        
    return node_temp, obj # type: ignore


def atingiuObj(tree: Tree) -> bool:
    """Verifica se a árvore já atingiu o objetivo

    Parameters
    ----------
    tree : Tree
        Nó raiz da árvore

    Returns
    -------
    bool
        Flag se atingiu objetivo

    """
    
    top_filho = tree.melhor_filho()
    
    return top_filho.eh_obj #type: ignore

  
def main():  
    # Se devemos mostrar a tela do jogo (+ lento) ou não (+ rápido)
    mostrar = True
 
    # Gera a árvore com o estado inicial do jogo 
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1', players=1)   # type: ignore
    env.reset()
    estado, x, _ = getState(getRam(env), raio)   # type: ignore
    tree = Tree(estado = estado, x = x, g = 0, action = 'None')

    # Se já existe alguma árvore, carrega
    if os.path.exists('AstarTree.pkl'):
        tree = pickle.load(open('AstarTree.pkl', 'rb'))

    # Repete enquanto não atingir objetivo    
    obj  = atingiuObj(tree)
    #obj = True
    while not obj:
        tree, obj = expande(tree, env, mostrar)

        # Grava estado atual da árvore por segurança
        fw = open('AstarTree.pkl', 'wb')
        pickle.dump(tree, fw)
        fw.close()
        
    # // todo implementar o retorno da melhor rota 
    #? descobrir se tá funcionando
    mostrar    = True
    acoes = tree.rota()
    emula([moves[acao] for acao in acoes], env, mostrar)
    
if __name__ == "__main__":
  main()