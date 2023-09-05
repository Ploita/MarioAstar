from __future__ import annotations
#!/usr/bin/env python
# marioAstar.py
# Author: Fabrício Olivetti de França
#
# A* algorithm for Super Mario World
# using RLE

import sys
import os
import pickle
import retro


#from typing import 
from rominfo import *
from utils import *

sys.setrecursionlimit(10000)

# quais movimentos estarão disponíveis
moves = {'direita':128, 'corre':130, 'pula':131, 'spin':386, 'esquerda':64}
translate = {0: 128, 1: 130, 2: 131, 3: 386, 4: 64}

# raio de visão (quadriculado raio x raio em torno do Mario)
raio = 6

# todo Separar as classes Tree e Nó
class Tree:
    """
        Classe da árvore de jogos para o Super Mario World
    """

    def __init__(self, estado: str, filhos: list[Tree] | None = None, pai: Tree | None = None, g: int = 0,
                  h: int = 0, terminal: bool = False, obj: bool = False, contador: int = 1):
        """Inicialização do nó/árvore

        Parameters
        ----------
        estado : str
            Representação N x N da tela do jogo, onde N é o raio de visão
        filhos : list[Tree] | None, optional
            Lista de nós filhos deste nó, by default None
        pai : Tree | None, optional
            Nó pai deste nó , by default None
        g : int, optional
            Valor de custo do Nó inicial até o Nó atual, pode ser lido como profundidade da árvore, by default 0
        h : int, optional
            Valor da heurística , by default 0
        terminal : bool, optional
            Indicador se o Nó atual é terminal, by default False
        obj : bool, optional
            Indicador se o Nó atual atingiu o objetivo, by default False
        contador : int, optional
            Numeração dos nós da árvore, by default 1
        """

        self.estado = estado
        self.filhos = filhos 
        
        self.g = g
        self.h = h
        
        self.eh_terminal = terminal
        self.eh_obj      = obj
        
        self.contador = contador
        self.pai = pai

    def __str__(self):
        return f"Tree(data={self.contador})"
  
  
def melhor_filho(tree: Tree) -> Tree | None:
    '''
    Encontra o melhor filho do nós representado por tree.
    
    Entrada: tree, o nó atual da árvore
    Saída:   a tupla (t, f) com t sendo o melhor filho de tree e f a heurística g+h
             retorna None caso o nó seja terminal
    '''
    # Implemente as tarefas abaixo e remove a instrução pass.
    # 1) Se o nó é terminal, retorna None
    if tree.eh_terminal:
        return None
    
    # 2) Se o nó não tem filhos, retorna ele mesmo e seu f
    if tree.filhos is None:
        return tree

    # 3) Para cada filho de tree, aplica melhor_filho e filtra aqueles que resultarem em None
    lista_filhos: list[Tree] = []
    for filho in tree.filhos:
        no_filho = melhor_filho(filho) 
        if no_filho is not None:
            lista_filhos.append(no_filho) 

    # 4) Se todos os filhos resultarem em terminal, marca tree como terminal e retorna None
    if not lista_filhos:
        tree.eh_terminal = True
        return None

    # 5) Caso contrário retorna aquele com o menor f
    lista_f = [[filho.g + filho.h, filho] for filho in lista_filhos]
    
    melhor_f: Tree
    melhor_f =  min(lista_f, key = lambda x: x[0])[1] # type: ignore
    # ? Tem um bug de tipo aqui acim que não soube ajeitar
    
    return melhor_f


# Nossa heurística é a quantidade
# de passos mínimos estimados para
# chegar ao final da fase
def heuristica(estado: str, x: int) -> int:
#    return (4800 - x)/8
    estNum = np.reshape(list(map(int, estado.split(','))), (2*raio+1,2*raio+1))
    dist = np.abs(estNum[:raio+1,raio+2:raio+7]).sum()
    return ((4800 - x)/8) + 0.3*dist
 
# Verifica se chegamos ao final   
def checaObj(estado, x):
    return x>4800

# Verifica se um nó é uma folha 
def folha(tree):
    """ Verifica se tree é um nó folha. """
    # Um nó folha é aquele que não tem filhos.
    if tree.filhos is None:
        return True
    return False

# Joga uma partida usando uma
# sequência de ações
def emula(acoes, env, mostrar):

    env.reset()

    while len(acoes)>0 and (not env.data.is_done()):
        a = acoes.pop(0)
        estado, xn, y = getState(getRam(env), raio)
        performAction(a, env)
        if mostrar:
            env.render()
    over = False
    estado, x, y = getState(getRam(env), raio)
    if env.data.is_done() or y > 400: 
        #Por algum motivo, o emulador não consegue percerber que quando o Mario
        #está abaixo do solo o jogo deveria acabar, e ele continua rodando ad infinitum
        over = True
    return estado, x, over
    
# Expande a árvore utilizando a heurística
def expande(tree, env, mostrar):
    '''Expande a árvore utilizando a heurística.
    
    Entrada: o nó raiz, o ambiente do retro Gym, booleano se devemos mostrar ou não a tela do jogo
    Saída:   a própria raiz E se atingiu o objetivo
    '''
    
    acoes = []

    # Se a árvore já for um nó folha
    # não tem ações a serem feitas 
    if folha(tree):
        raiz  = tree
        filho = tree
    else:

        # Busca pelo melhor nó folha
        filho = melhor_filho(tree)
        
        # Retorna para a raiz gravando as ações efetuadas
        raiz = filho

        # 1) Enquanto o pai de raiz não for None
        while raiz.pai is not None:
        
        # 2) Atribua raiz a uma variável neto
            neto = raiz

        # 3) faça raiz = seu próprio pai
            raiz = raiz.pai

        # 4) verifique qual a ação de raiz leva ao nó neto
            for k,v in enumerate(moves.items()):
                temp1 = raiz.filhos[k]
                if temp1.g + temp1.h == neto.g + neto.h:
                    acoes.append(v[0])
        # 5) faça um append dessa ação na lista acoes

        
        # inverte a lista de ações e imprime para debug
    acoes.reverse()
    print('ACOES:  (  ', filho.g, ' ): ',  acoes[200:])
    
    # Vamos assumir que não atingiu o objetivo
    obj = False

    # Gera cada um dos filhos e verifica se atingiu objetivo
    filho.filhos = []
    maxX         = 0
    for k, v in moves.items():
        estado, x, over = emula([moves[acao] for acao in acoes] + [v], env, mostrar)
        maxX            = max(x, maxX)
        if obj or checaObj(estado, x):
            obj = True
            over = True
        filho.filhos.append(Tree(estado, g=filho.g + 1, h=heuristica(estado,x),
                                    pai=filho, terminal=over, obj=obj))
    print('FALTA: ', heuristica(estado, maxX))
        
    return raiz, obj

# Verifica se a árvore já atingiu o objetivo
def atingiuObj(tree):
    ''' Verifica se atingiu o objetivo 
    
    Entrada: um nó da árvore
    Saída:   (True, acoes) se atingiu o objetivo, sendo acoes a sequência de ações para chegar até ele.
             (False, [])  se não atingiu o objetivo
    '''
    
    # Complete as tarefas a seguir e remova a instrução pass
    
    # 1) Se o nó é terminal, retorna o valor de eh_obj e a lista vazia de ações

    if tree.eh_terminal:
        return tree.eh_obj, []
    
    # 2) Se o conjunto de filhos é None, retorna falso e lista vazia, pois não atingiu o obj
    
    if tree.filhos is None:
        return False, []
    
    # 3) Se nenhum dos anteriores retornou, para cada movimento "k" e valor "v" possível do dicionário moves:
    #       chama recursivamente atingiuObj com o filho do movimento "k" e recebe obj, acoes
    #       Se obj for verdadeiro, retorna obj e a lista de acoes concatenado com "v"
    for index, filho in enumerate(tree.filhos):
        obj, acoes = atingiuObj(filho)
        if obj == True:
            return obj, [translate[index]] + acoes
    
    # 4) Se chegar ao final do laço sem retorna, retorne falso e vazio
    return False, []


# Gera a árvore utilizando A*
def astar():
    
    # Se devemos mostrar a tela do jogo (+ lento) ou não (+ rápido)
    mostrar = 0
 
    # Gera a árvore com o estado inicial do jogo 
    env = retro.make(game='SuperMarioWorld-Snes', state='DonutPlains1', players=1)    
    env.reset()
    estado, x, _ = getState(getRam(env), raio)  
    tree         = Tree(estado, g=0, h=heuristica(estado,x))

    # Se já existe alguma árvore, carrega
    if os.path.exists('AstarTree.pkl'):
        tree = pickle.load(open('AstarTree.pkl', 'rb'))

    # Repete enquanto não atingir objetivo    
     obj, acoes  = atingiuObj(tree)

    while not obj:
        tree, obj = expande(tree, env, mostrar)

        # Grava estado atual da árvore por segurança
        fw = open('AstarTree.pkl', 'wb')
        pickle.dump(tree, fw)
        fw.close()
        
    obj, acoes = atingiuObj(tree)
    acoes.insert(0, 131)
    print(acoes)
    mostrar    = True
    emula(acoes, env, mostrar)

    return tree
  
def main():  
  tree = astar()
    
if __name__ == "__main__":
  main()