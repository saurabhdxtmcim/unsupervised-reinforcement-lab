# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

"""
Deep Q-Learning Explained Simply
================================

This file explains Deep Q-Learning concepts with simple examples and visualizations
to help you understand what's happening under the hood.

What is Deep Q-Learning?
========================

Think of Deep Q-Learning like teaching a video game player to get better at a game:

1. The player (agent) observes the game screen (state)
2. The player decides what button to press (action)  
3. The game gives a score/penalty (reward)
4. The player learns from this experience to make better decisions next time

The "Deep" part means we use a neural network to help the player remember
and make smart decisions based on what they've learned.

Key Concepts Explained with Analogies:
=====================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def explain_q_values():
    """
    Explain what Q-values represent with a simple example.
    """
    print("=" * 60)
    print("1. WHAT ARE Q-VALUES?")
    print("=" * 60)
    
    print("""
Q-values are like a "quality score" for each action in each situation.

Imagine you're playing a video game and you're at a crossroads:
- State: "Standing at crossroads with 100 health points"
- Actions: [Go Left, Go Right, Go Forward, Go Back]
- Q-values: [50, 85, 20, 30]

This means:
- Going Right (Q=85) is probably the BEST action
- Going Left (Q=50) is okay  
- Going Forward (Q=20) or Back (Q=30) are not great choices

The agent will usually pick the action with the highest Q-value!
    """)

def explain_neural_network_role():
    """
    Explain why we need a neural network for Q-Learning.
    """
    print("=" * 60)
    print("2. WHY DO WE NEED A NEURAL NETWORK?")
    print("=" * 60)
    
    print("""
Traditional Q-Learning uses a table to store Q-values:

Simple Game (like Tic-Tac-Toe):
State 1: [Action1: 10, Action2: 5, Action3: 8]
State 2: [Action1: 3,  Action2: 12, Action3: 7]
...and so on

But for complex games (like CartPole):
- States are continuous numbers (cart position: 2.34, velocity: -1.67, etc.)
- We can't make a table for EVERY possible state combination!
- There would be infinite rows in our table!

Solution: Neural Network!
Instead of a giant table, we train a neural network to predict:
"Given this state, what are the Q-values for each action?"

Input: [cart_position, cart_velocity, pole_angle, pole_velocity]
Output: [Q_value_left, Q_value_right]
    """)

def create_q_learning_visualization():
    """
    Create a visual representation of the Q-Learning process.
    """
    print("=" * 60)
    print("3. THE DEEP Q-LEARNING PROCESS")
    print("=" * 60)
    
    # Create a figure showing the learning process
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Agent observes state
    ax1.set_title("Step 1: Agent Observes State", fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.7, "Current State:", ha='center', fontsize=12, fontweight='bold')
    ax1.text(0.5, 0.5, "Cart Position: 0.1", ha='center', fontsize=10)
    ax1.text(0.5, 0.4, "Cart Velocity: -0.5", ha='center', fontsize=10)
    ax1.text(0.5, 0.3, "Pole Angle: 0.02", ha='center', fontsize=10)
    ax1.text(0.5, 0.2, "Pole Velocity: 0.8", ha='center', fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. Neural network predicts Q-values
    ax2.set_title("Step 2: Neural Network Predicts Q-values", fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.7, "Q-Network Prediction:", ha='center', fontsize=12, fontweight='bold')
    ax2.text(0.5, 0.5, "Q(Left) = 0.65", ha='center', fontsize=11, color='blue')
    ax2.text(0.5, 0.3, "Q(Right) = 0.82 ← Best!", ha='center', fontsize=11, color='red', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 3. Agent takes action and gets reward
    ax3.set_title("Step 3: Take Action & Get Reward", fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.7, "Action Taken: Right", ha='center', fontsize=12, fontweight='bold')
    ax3.text(0.5, 0.5, "Reward Received: +1", ha='center', fontsize=11, color='green')
    ax3.text(0.5, 0.3, "New State: [0.15, -0.2, 0.01, 0.3]", ha='center', fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # 4. Learning and improvement
    ax4.set_title("Step 4: Learn and Improve", fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.7, "Learning Process:", ha='center', fontsize=12, fontweight='bold')
    ax4.text(0.5, 0.5, "✓ Store experience in memory", ha='center', fontsize=10)
    ax4.text(0.5, 0.4, "✓ Update neural network weights", ha='center', fontsize=10)
    ax4.text(0.5, 0.3, "✓ Improve future predictions", ha='center', fontsize=10)
    ax4.text(0.5, 0.1, "Repeat for next state!", ha='center', fontsize=11, color='purple', fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('deep_q_learning_process.png', dpi=150, bbox_inches='tight')
    print("Deep Q-Learning process diagram saved as: deep_q_learning_process.png")

def explain_experience_replay():
    """
    Explain experience replay with a simple analogy.
    """
    print("=" * 60)
    print("4. EXPERIENCE REPLAY - Learning from the Past")
    print("=" * 60)
    
    print("""
Experience Replay is like keeping a diary of your gaming experiences!

Without Experience Replay (Bad):
- You only learn from your most recent game experience
- Like studying only the last math problem you solved
- Very inefficient and unstable learning

With Experience Replay (Good):
- Keep a "memory buffer" of past experiences
- Randomly sample old experiences to learn from
- Like reviewing your old math homework randomly to reinforce learning

Example Memory Buffer:
Experience 1: State=[1,2,3,4], Action=Right, Reward=+1, Next_State=[1.1,2.2,3.3,4.4]
Experience 2: State=[5,6,7,8], Action=Left,  Reward=+1, Next_State=[4.9,5.8,6.7,7.6]
Experience 3: State=[2,3,1,5], Action=Right, Reward=-1, Next_State=[2.1,3.1,1.1,5.1]
...and thousands more!

During training:
- Randomly pick 64 experiences from memory
- Learn from this diverse batch
- Much more stable learning!
    """)

def explain_target_network():
    """
    Explain target network concept.
    """
    print("=" * 60)
    print("5. TARGET NETWORK - Stable Learning")
    print("=" * 60)
    
    print("""
Target Network is like having a "stable reference point" for learning.

Problem without Target Network:
Imagine you're trying to hit a moving target while the target keeps
moving based on where you're aiming! Very confusing and unstable.

Solution with Target Network:
- Main Network: Learns and updates frequently (like your current aim)
- Target Network: Stays stable for longer periods (like a steady target)
- Every few steps, copy main network weights to target network

This gives the agent a stable "target" to learn towards, making
training much more stable and reliable.

Think of it like:
- Main Network: "What I think right now"
- Target Network: "What I thought was good recently"
    """)

def create_training_progress_example():
    """
    Show what training progress looks like.
    """
    print("=" * 60)
    print("6. WHAT DOES TRAINING LOOK like?")
    print("=" * 60)
    
    # Simulate training progress
    episodes = np.arange(1, 1001)
    
    # Simulate realistic learning curve
    np.random.seed(42)
    base_performance = 50 + 150 * (1 - np.exp(-episodes / 200))
    noise = np.random.normal(0, 20, len(episodes))
    scores = np.maximum(10, base_performance + noise)
    
    # Add some episodes where agent fails early
    for i in range(0, 200, 20):
        if i < len(scores):
            scores[i] = np.random.uniform(10, 30)
    
    plt.figure(figsize=(14, 8))
    
    # Plot individual episode scores
    plt.subplot(2, 2, 1)
    plt.plot(episodes[:200], scores[:200], alpha=0.7, linewidth=1, color='lightblue')
    plt.title('Early Training (Episodes 1-200)\nLots of Random Exploration', fontsize=12)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(episodes[200:500], scores[200:500], alpha=0.7, linewidth=1, color='orange')
    plt.title('Mid Training (Episodes 200-500)\nLearning Patterns', fontsize=12)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(episodes[500:], scores[500:], alpha=0.7, linewidth=1, color='green')
    plt.title('Late Training (Episodes 500-1000)\nConsistent Performance', fontsize=12)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    # Full training curve with moving average
    plt.subplot(2, 2, 4)
    plt.plot(episodes, scores, alpha=0.3, linewidth=1, color='blue', label='Episode Scores')
    
    # Calculate moving average
    window = 100
    moving_avg = []
    for i in range(window-1, len(scores)):
        moving_avg.append(np.mean(scores[i-window+1:i+1]))
    
    plt.plot(episodes[window-1:], moving_avg, color='red', linewidth=3, label='100-Episode Average')
    plt.axhline(y=195, color='green', linestyle='--', linewidth=2, label='Solved Threshold')
    plt.title('Complete Training Progress', fontsize=12)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress_example.png', dpi=150, bbox_inches='tight')
    print("Training progress example saved as: training_progress_example.png")
    
    print("""
What you see in training:

Early Episodes (1-200):
- Agent acts randomly (high exploration)
- Scores are low and inconsistent
- Learning basic patterns

Mid Episodes (200-500):  
- Agent starts finding good strategies
- Scores gradually improve
- Still some exploration happening

Late Episodes (500-1000):
- Agent becomes more consistent
- High scores most of the time
- Mostly exploitation, little exploration

Success Criteria:
- Average score ≥ 195 over 100 consecutive episodes
- Shows the agent has truly "learned" the task
    """)

def explain_hyperparameters():
    """
    Explain key hyperparameters and their effects.
    """
    print("=" * 60)
    print("7. KEY HYPERPARAMETERS - The Knobs You Can Tune")
    print("=" * 60)
    
    print("""
Learning Rate (α = 0.001):
- How fast the neural network learns
- Too high: Learning is unstable, jumps around
- Too low: Learning is very slow
- Think: How big steps to take when climbing a hill

Discount Factor (γ = 0.995):
- How much the agent cares about future rewards
- 0.9: Cares more about immediate rewards
- 0.99: Cares a lot about future rewards
- 0.999: Very long-term thinking

Epsilon (ε-greedy exploration):
- Start: 1.0 (100% random actions - pure exploration)
- End: 0.01 (1% random actions - mostly exploitation)
- Decay: 0.995 (gradually reduce exploration)
- Think: Balance between trying new things vs using what you know

Memory Buffer Size (100,000):
- How many past experiences to remember
- Larger: More diverse learning, but uses more memory
- Smaller: Faster, but less diverse experiences

Batch Size (64):
- How many experiences to learn from at once
- Larger: More stable learning, but slower
- Smaller: Faster updates, but more noisy learning

Update Frequency (every 4 steps):
- How often to train the neural network
- More frequent: Faster learning, but less stable
- Less frequent: More stable, but slower learning
    """)

def main():
    """
    Run the complete Deep Q-Learning explanation.
    """
    print("🎮 DEEP Q-LEARNING EXPLAINED SIMPLY 🎮")
    print("Understanding AI that learns to play games like humans!")
    print()
    
    explain_q_values()
    explain_neural_network_role()
    
    print("Creating visualizations...")
    create_q_learning_visualization()
    
    explain_experience_replay()
    explain_target_network()
    
    print("Creating training progress example...")
    create_training_progress_example()
    
    explain_hyperparameters()
    
    print("=" * 60)
    print("8. PUTTING IT ALL TOGETHER")
    print("=" * 60)
    
    print("""
Deep Q-Learning Recipe:
======================

1. 🧠 Neural Network: Predicts Q-values for each action
2. 🎯 Target Network: Provides stable learning targets  
3. 💾 Experience Replay: Learns from diverse past experiences
4. 🎲 ε-greedy Policy: Balances exploration vs exploitation
5. 📈 Training Loop: Gradually improves over many episodes

The Magic:
- Agent starts knowing nothing (random actions)
- Gradually learns which actions lead to good outcomes
- Eventually becomes an expert at the task!

Real-world Applications:
- Game playing (Chess, Go, Video games)
- Robot control (Walking, Manipulation)
- Trading and finance
- Resource management
- Autonomous vehicles

The key insight: Instead of programming rules, we let the AI
discover the best strategy through trial and error, just like
how humans learn!
    """)
    
    print("=" * 60)
    print("✅ EXPLANATION COMPLETE!")
    print("Check the generated images for visual explanations:")
    print("- deep_q_learning_process.png")
    print("- training_progress_example.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
