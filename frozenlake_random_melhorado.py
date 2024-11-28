import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle
import shutil
import os
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import streamlit as st


class ReiforcementLearningAgent:
    def __init__(self,qtable,learnig_rate,discount_factor,epsilon,env):
        self.qtable=defaultdict(lambda: np.zeros(env.action_space.n),qtable)
        self.learning_rate=learnig_rate
        self.discount_factor=discount_factor
        self.epsilon=epsilon

    def decisaoTreino(self,estado,env:gym.Env)->any:
        if(np.random.rand() > self.epsilon):
            acao=env.action_space.sample()
        else:
            acao=np.random.choice(np.where(self.qtable[estado]==self.qtable[estado].max())[0])
        return acao

    def decisaoTeste(self,estado)->any:
        acao=np.random.choice(np.where(self.qtable[estado]==self.qtable[estado].max())[0])
        return acao
    
    def atualizar(self,estado,proximo_estado,recompensa,acao,terminou):
        maior_proximo= max(self.qtable[proximo_estado]) * (not terminou)
        atual=self.qtable[estado][acao]
        self.qtable[estado][acao]=atual + self.learning_rate*(recompensa + self.discount_factor*maior_proximo-atual)


class FrozenLake:
    def __init__(self):
        self.learning_rate=0.8
        self.discount_factor=0.95
        self.epsilon=0.1
        self.size=4
        self.frozen=0.6
        self.treino_mapa=1
        self.num_mapa=10000
        self.namefile='qtablemelhor.pkl'
        self.backupfile='qtablemelhor_backup.pkl'

    def abrirArquivo(self):
        try:
            with open(self.namefile, 'rb') as arquivo:
                carregar = pickle.load(arquivo)
        except (FileNotFoundError, EOFError):
            with open(self.namefile,'wb') as arquivo:
                pass
            carregar = {}
        return carregar

    def salvarArquivo(self,qtable):
        if os.path.exists(self.namefile):
            shutil.copy(self.namefile, self.backupfile)  # Faz um backup do arquivo existente
        with open('temp.pkl', 'wb') as arquivo:
            pickle.dump(dict(qtable), arquivo)
        if os.path.exists(self.namefile):
            os.remove(self.namefile)
        os.rename('temp.pkl',self.namefile)

    def treinarMapa(self,robo:ReiforcementLearningAgent,env:gym.Env,mapa):
        global img_placeholder
        estado, info = env.reset()
        estado=tuple(mapa),estado
        episode_over=False
        while not episode_over:
            imagem = env.render()  
            img_placeholder.image(imagem, channels="RGB", use_column_width=True)
            acao = robo.decisaoTreino(estado=estado,env=env)
            proximo_estado, recompensa, terminou, truncou, p = env.step(acao)
            if recompensa == 0 and terminou == True:
                recompensa = -1
            robo.atualizar(estado=estado,proximo_estado=proximo_estado,recompensa=recompensa,acao=acao,terminou=terminou)
            estado=proximo_estado
            self.salvarArquivo(robo.qtable)
            episode_over = terminou or truncou
    


            
    def testarMapa(self,robo:ReiforcementLearningAgent,env:gym.Env,mapa):
        estado, info = env.reset()
        estado = tuple(mapa),estado
        episode_over=False
        global img_placeholder
        while not episode_over:
            imagem = env.render()  
            img_placeholder.image(imagem, channels="RGB", use_column_width=True)
            acao = robo.decisaoTeste(estado=estado)
            proximo_estado, recompensa, terminou, truncou, p = env.step(acao)
            estado=proximo_estado
            episode_over = terminou or truncou


    def mapa(self,robo:ReiforcementLearningAgent):
        mapa=generate_random_map(size=self.size,p=self.frozen)
        env = gym.make('FrozenLake-v1', is_slippery=False,render_mode="rgb_array",desc=mapa)
        for map in range(self.treino_mapa):
            self.testarMapa(robo=robo,env=env,mapa=mapa)
        #input("Pressione para testar se o robo aprendeu o mapa")
        self.testarMapa(robo=robo,env=env,mapa=mapa)
        env.close()

    def testarRobo(self,robo:ReiforcementLearningAgent):
        mapa=generate_random_map(size=self.size,p=self.frozen)
        env = gym.make('FrozenLake-v1', is_slippery=False,render_mode="rgb_array",desc=mapa)
        #input("Pressione para testar se o robo aprendeu o jogo de maneira generica")
        self.testarMapa(robo=robo,env=env,mapa=mapa)
        env.close

    def frozenLake(self):
        env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')
        robo = ReiforcementLearningAgent(qtable=self.abrirArquivo(),learnig_rate=self.learning_rate,discount_factor=self.discount_factor,epsilon=self.epsilon,env=env)
        env.close()
        for episodio in range(self.num_mapa):
            self.mapa(robo)
        
img_placeholder = st.empty()  # Cria o placeholder para a imagem   
FrozenLake().frozenLake()
