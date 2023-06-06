class Tree:
    def __init__(self, estado, filhos=None, pai=None, g=0, h=0, terminal=False, obj=False):
        self.estado   = estado
        self.filhos   = filhos # lista de filhos desse nó
        
        self.g = g
        self.h = h
        
        self.eh_terminal = terminal
        self.eh_obj      = obj
        
        self.pai = pai # apontador para o pai, útil para fazer o backtracking

    def __str__(self):
        return self.estado
    
def melhor_filho(tree):
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
        return tree, tree.g + tree.h

    # 3) Para cada filho de tree, aplica melhor_filho e filtra aqueles que resultarem em None
    lista_filhos = []
    for filho in tree.filhos:
        no_filho = melhor_filho(filho) 
        if no_filho is not None:
            lista_filhos.append(no_filho[0])

    # 4) Se todos os filhos resultarem em terminal, marca tree como terminal e retorna None
    if not lista_filhos:
        tree.eh_terminal = True
        return None

    # 5) Caso contrário retorna aquele com o menor f
    lista_f = []
    for index, filho in enumerate(lista_filhos):
        lista_f.append([filho.g + filho.h, index])
    
    #Para voltar a proposta original, delete este if
    if tree.g > 300:
        melhor_f = min(lista_f)
        candidatos = [arvore for arvore in lista_f if arvore[0]== melhor_f[0]]
        escolhido = choice(range(len(candidatos)))
        return lista_filhos[escolhido], lista_filhos[escolhido].g + lista_filhos[escolhido].h
     
    melhor_filho_f = min(lista_f)
    return lista_filhos[melhor_filho_f[1]], melhor_filho_f[0]

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
    for k, v in enumerate(tree.filhos):
        obj, acoes = atingiuObj(v)
        if obj == True:
            return obj, acoes + [k]
    
    # 4) Se chegar ao final do laço sem retorna, retorne falso e vazio
    return False, []

P = Tree('P', None,None,6,0,True,True)
O = Tree('O', [P],None,5,1,False,False)
N = Tree('N', None,None,5,1,True,False)
M = Tree('M', [O],None,4,2,False,False)
L = Tree('L', [N],None,4,2,False,False)
K = Tree('K', [M],None,3,3,False,False)
J = Tree('J', None,None,3,3,False,False)
I = Tree('I', None,None,3,3,False,False)
H = Tree('H', [L],None,3,3,False,False)
G = Tree('G', [K],None,2,4,False,False)
F = Tree('F', None,None,2,4,False,False)
E = Tree('E', [I, J],None,2,4,False,False)
D = Tree('D', [H],None,2,4,False,False)
C = Tree('C', [F, G],None,1,5,False,False)
B = Tree('B', [D, E],None,1,5,False,False)
A = Tree('A', [B, C],None,0,6,False,False)

_, temp = atingiuObj(A)
temp.reverse()
temp = 1

import retro

movie = retro.Movie('SuperMarioWorld-Snes-YoshiIsland1-000001.bk2')
movie.step()

env = retro.make(
    game=movie.get_game(),
    state=None,
    # bk2s can contain any button presses, so allow everything
    use_restricted_actions=retro.Actions.ALL,
    players=movie.players,
)
env.initial_state = movie.get_state()
env.reset()
keys = []
while movie.step():
    
    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))
    env.step(keys)

print(keys)