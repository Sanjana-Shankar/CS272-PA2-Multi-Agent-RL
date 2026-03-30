from __future__ import annotations
#from mycheckersenv import env
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# =========================
# Core game constants
# =========================

# Board encoding
EMPTY = 0

P0_MAN = 1
P0_KING = 2

P1_MAN = -1
P1_KING = -2

BOARD_SIZE = 6
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
ACTION_SIZE = NUM_SQUARES * NUM_SQUARES

Position = Tuple[int, int]

# Decorator that automatically generates __init__ for classes
# Reduces boilerplate code for classes that primarily store data
# frozen=True: Makes instances of the dataclass immutable, once an object
# of the dataclass is created, its attributes cannot be changed.
# Any attempt to modify an attribute will raise FrozenInstanceError
# Useful for creating objects whose state should not change after creation, ensuring
# data integrity and making them safe to use as dictionary keys or in sets
@dataclass(frozen=True)
class Move:
  start: Position
  end: Position
  captured: Optional[Position] = None

  # Define methods in a class that can be accessed like attributes without explicitly
  # calling them as functions
  # Its commonly used to provide a "getter" method for an attribute
  # add logic like validation, computation, or formatting, when an attribute is accessed,
  # while still maintaining a clean, attribute-like interface for the user of the class
  @property
  def is_capture(self) -> bool:
    return self.captured is not None
class Checkers6x6:
  '''
  Includes:
  - board initialization
  - normal movement
  - single capture moves
  - king promotion
  - winner detection
  '''
  def __init__(self) -> None:
    self.board: List[List[int]] = self._create_initial_board()
    self.current_player: int = 0 # 0 or 1
    self.winner: Optional[int] = None

    # If a player is in the middle of a multi-jump, they must continue
    # with this piece only.
    self.forced_piece: Optional[Position] = None

  def _create_initial_board(self) -> List[List[int]]:
    '''
    Standard 6x6 checkers setup:
    - Player 1 pieces on top two rows (negative values)
    - Player 0 pieces on bottom two rows (positive values)
    - Pieces placed on dark squares only
    '''
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    # Player 1 starts at top
    for row in range(2):
      for col in range(BOARD_SIZE):
        if self._is_dark_square(row, col):
          board[row][col] = P1_MAN

    # Player 0 starts at bottom
    for row in range(BOARD_SIZE-2, BOARD_SIZE):
      for col in range(BOARD_SIZE):
        if self._is_dark_square(row, col):
          board[row][col] = P0_MAN
    return board

  @staticmethod
  def _is_dark_square(row: int, col: int) -> bool:
    return (row + col) % 2 == 1

  @staticmethod
  def _in_bounds(row: int, col: int) -> bool:
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
      return True
    return False

  @staticmethod
  def _belongs_to_player(piece: int, player: int) -> bool:
    if player == 0:
      return piece in (P0_MAN, P0_KING)
    return piece in (P1_MAN, P1_KING)

  @staticmethod
  def _is_opponent_piece(piece: int, player: int) -> bool:
    if(player == 0):
      return piece in (P1_MAN, P1_KING)
    return piece in (P0_MAN, P0_KING)

  @staticmethod
  def _is_king(piece: int) -> bool:
    return piece in (P0_KING, P1_KING)

  def reset(self) -> None:
    self.board = self._create_initial_board()
    self.current_player = 0
    self.winner = None
    self.forced_piece = None

  def print_board(self) -> None:
    '''
    Simple text rendering
    '''
    piece_to_char = {
        EMPTY: ".",
        P0_MAN: "r",
        P0_KING: "R",
        P1_MAN: "b",
        P1_KING: "B",
    }

    print("  " + " ".join(str(c) for c in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
      row_str = " ".join(piece_to_char[self.board[r][c]] for c in range(BOARD_SIZE))
      print(f"{r} {row_str}")
    print(f"Current player: player_{self.current_player}")
    print(f"Forced piece: {self.forced_piece}")
    if self.winner is not None:
      print(f"Winner: player_{self.winner}")
    print()

  def get_piece_directions(self, piece: int) -> List[Tuple[int, int]]:
    '''
    Returns legal diagonal step directions for the piece type
    Player 0 moves upward (-1 row).
    Player 1 moves downward (+1 row).
    Kings move both directions.
    '''
    if piece == P0_MAN:
      return [(-1, -1), (-1, 1)] # Bottom player so can only move diagonally upward
    if piece == P1_MAN:
      return [(1, -1), (1, 1)] # Top player so can only move diagonally downward
    if self._is_king(piece):
      return [(-1, -1), (-1, 1), (1, -1), (1, 1)] # Can move any direction diagonally up or down

  def get_normal_moves_for_piece(self, row: int, col: int) -> List[Move]:
    if not self._in_bounds(row, col):
      return []

    piece = self.board[row][col]
    if piece == EMPTY:
      return []

    if not self._belongs_to_player(piece, self.current_player):
      return []

    moves: List[Move] = []
    for dr, dc in self.get_piece_directions(piece):
      nr, nc = row + dr, col + dc
      if self._in_bounds(nr, nc) and self.board[nr][nc] == EMPTY:
        moves.append(Move(start=(row, col), end=(nr, nc)))
    return moves

  def get_capture_moves_for_piece(self, row: int, col: int) -> List[Move]:
    if not self._in_bounds(row, col):
      return []

    piece = self.board[row][col]
    if piece == EMPTY:
      return []

    if not self._belongs_to_player(piece, self.current_player):
      return []

    captures: List[Move] = []
    for dr, dc in self.get_piece_directions(piece):
      mid_r, mid_c = row + dr, col + dc
      jump_r, jump_c = row + 2 * dr, col + 2 * dc

      if(
          self._in_bounds(mid_r, mid_c)
          and self._in_bounds(jump_r, jump_c)
          and self._is_opponent_piece(self.board[mid_r][mid_c], self.current_player)
          and self.board[jump_r][jump_c] == EMPTY
      ):
        captures.append(
            Move (
              start=(row, col),
              end=(jump_r, jump_c),
              captured=(mid_r, mid_c),
            )
        )
    return captures

  def _get_all_capture_moves(self) -> List[Move]:
    captures: List[Move] = []

    if self.forced_piece is not None:
      r, c = self.forced_piece
      return self.get_capture_moves_for_piece(r, c)

    for r in range(BOARD_SIZE):
      for c in range(BOARD_SIZE):
        captures.extend(self.get_capture_moves_for_piece(r, c))
    return captures

  def _get_all_normal_moves(self) -> List[Move]:
    moves: List[Move] = []

    # During a multi-jump continuation, normal moves are not allowed.
    if self.forced_piece is not None:
      return []

    for r in range(BOARD_SIZE):
      for c in range(BOARD_SIZE):
        moves.extend(self.get_normal_moves_for_piece(r, c))
    return moves

  def get_all_legal_moves(self) -> List[Move]:
    '''
    Mandatory capture rule:
    - If any capture exists, only capture moves are legal
    - If in a multi-jump chain, only capture moves for forced_piece are legal
    '''
    captures = self._get_all_capture_moves()
    if captures:
      return captures
    return self._get_all_normal_moves()


  def apply_move(self, move: Move) -> bool:
      '''
      Applies a move if legal, updates the board, promotes kings,
      checks for winner, and switches turns.
      - Mandatory captures enforced
      - After a capture, if another capture is available from the landing square,
      the same player continues with that same piece
      - otherwise turn switches
      '''
      if self.winner is not None:
        return False

      legal_moves = self.get_all_legal_moves()
      if move not in legal_moves:
        return False

      sr, sc = move.start
      er, ec = move.end

      piece = self.board[sr][sc]
      self.board[sr][sc] = EMPTY
      self.board[er][ec] = piece

      if move.is_capture and move.captured is not None:
        cr, cc = move.captured
        self.board[cr][cc] = EMPTY

      self._promote_if_needed(er, ec)
      self._update_winner()

      if self.winner is not None:
        return True

      # Multi-jump continuation
      if move.is_capture:
        further_captures = self.get_capture_moves_for_piece(er, ec)
        if further_captures:
          self.forced_piece = (er, ec)
          return True

      # End turn
      self.forced_piece = None
      self.current_player = 1 - self.current_player
      self._update_winner()
      return True

  def _promote_if_needed(self, row: int, col: int) -> None:
      piece = self.board[row][col]

      if piece == P0_MAN and row == 0:
        self.board[row][col] = P0_KING
      elif piece == P1_MAN and row == BOARD_SIZE - 1:
        self.board[row][col] = P1_KING

  def _count_player_pieces(self, player: int) -> int:
      count = 0
      for row in self.board:
        for piece in row:
          if self._belongs_to_player(piece, player):
            count += 1
      return count

  def _player_has_any_moves(self, player: int) -> bool:
      saved_player = self.current_player
      saved_forced_piece = self.forced_piece

      self.current_player = player
      self.forced_piece = None
      moves_exist = len(self.get_all_legal_moves()) > 0

      self.current_player = saved_player
      self.forced_piece = saved_forced_piece
      return moves_exist

  def _update_winner(self) -> None:
      '''
      Win condition:
      - opponent has no pieces
      - oppoenent has no legal moves
      '''
      p0_count = self._count_player_pieces(0)
      p1_count = self._count_player_pieces(1)

      if p0_count == 0:
        self.winner = 1
        return
      if p1_count == 0:
        self.winner = 0
        return

      if not self._player_has_any_moves(0):
        self.winner = 1
        return
      if not self._player_has_any_moves(1):
        self.winner = 0
        return

      self.winner = None

# Phase 3 of Task 1: Petting Zoo AEC Environment
class Checkers6x6AECEnv(AECEnv):
  metadata = {
      "name": "checkers_6x6_v0",
      "render_modes": ["human"],
      "is_parallelizable": False,
  }

  def __init__(self, max_turns: int = 200, render_mode: Optional[str] = None):
    super().__init__()

    self.render_mode = render_mode
    self.max_turns = max_turns

    self.possible_agents = ["player_0", "player_1"]
    self.agent_name_mapping = {
        "player_0": 0,
        "player_1": 1,
    }

    self.engine = Checkers6x6()
    self.turn_count = 0

    self._action_spaces = {
        agent: spaces.Discrete(ACTION_SIZE) for agent in self.possible_agents
    }

    self._observation_spaces = {
        agent: spaces.Dict(
            {
                "Observation": spaces.Box(
                    low=-2,
                    high=2,
                    shape=(BOARD_SIZE, BOARD_SIZE),
                    dtype=np.int8,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(ACTION_SIZE,),
                    dtype=np.int8,
                ),
            }
        )
        for agent in self.possible_agents
    }


  def observation_space(self, agent: str):
    return self._observation_spaces[agent]

  def action_space(self, agent: str):
    return self._action_spaces[agent]

  def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    self.engine.reset()
    self.turn_count = 0

    self.agents = self.possible_agents[:]
    self.rewards = {agent: 0.0 for agent in self.agents}
    self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
    self.terminations = {agent: False for agent in self.agents}
    self.truncations = {agent: False for agent in self.agents}
    self.infos = {
        agent: {"legal_moves": []} for agent in self.agents
    }
    self._agent_selector = agent_selector.agent_selector(self.agents)
    self.agent_selection = self.possible_agents[self.engine.current_player]

    self._update_infos()

  def observe(self, agent: str) -> Dict[str, np.ndarray]:
    board = np.array(self.engine.board, dtype=np.int8);

    # Optional perspective flip:
    # current agent sees own pieces as positive
    player_idx = self.agent_name_mapping[agent]
    if player_idx == 1:
      board = -board

    action_mask = self._get_action_mask(agent)
    return {
        "Observation": board,
        "action_mask": action_mask,
    }

  def step(self, action: int) -> None:
    if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
      self._was_dead_step(action)
      return

    agent = self.agent_selection
    player_idx = self.agent_name_mapping[agent]

    if player_idx != self.engine.current_player:
      raise ValueError("It is not {agent} turn.")

    legal_moves  = self.engine.get_all_legal_moves()
    legal_actions = {self.encode_move(m): m for m in legal_moves}

    self.rewards = {a: 0.0 for a in self.agents}

    if action not in legal_actions:
      # Illegal action penalty
      self.rewards[agent] = -1.0
      other = self.possible_agents[1 - player_idx]
      self.rewards[other] = 1.0

      self.terminations = {a: True for a in self.agents}
      self._accumulate_rewards()
      return

    move = legal_actions[action]
    prev_player = self.engine.current_player
    applied = self.engine.apply_move(move)

    if not applied:
      raise RuntimeError("Legal move failed to apply unexpectedly.")

    self.turn_count += 1

    # Terminal rewards
    if self.engine.winner is not None:
      winner_agent = self.possible_agents[self.engine.winner]
      loser_agent = self.possible_agents[1 - self.engine.winner]
      self.rewards[winner_agent] = 1.0
      self.rewards[loser_agent] = -1.0
      self.terminations = {a: True for a in self.agents}

    elif self.turn_count >= self.max_turns:
      self.truncations = {a: True for a in self.agents}
    self._update_infos()

    # If multi-jump continues, same player goes again
    if not any(self.terminations.values()) and not any(self.truncations.values()):
      self.agent_selection = self.possible_agents[self.engine.current_player]

    self._accumulate_rewards()

    if self.render_mode == "human":
      self.render()

  def render(self) -> None:
    piece_to_char = {
        EMPTY: ".",
        P0_MAN: "r",
        P0_KING: "R",
        P1_MAN: "b",
        P1_KING: "B",
    }

    print(" " + " ".join(str(c) for c in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
      row_str = " ".join(piece_to_char[self.engine.board[r][c]] for c in range(BOARD_SIZE))
      print(f"{r} {row_str}")
    print(f"Current player: player_{self.engine.current_player}")
    print(f"Forced piece: {self.engine.forced_piece}")
    print(f"Turn count: {self.turn_count}")
    print()

  def close(self) -> None:
    pass

  def _update_infos(self) -> None:
    for agent in self.agents:
      if self.agent_name_mapping[agent] == self.engine.current_player:
        self.infos[agent]["legal_moves"] = [
            self.encode_move(m) for m in self.engine.get_all_legal_moves()
        ]
      else:
        self.infos[agent]["legal_moves"] = []

  def _get_action_mask(self, agent: str) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype= np.int8)

    if agent not in self.agents:
      return mask

    if self.terminations.get(agent, False) or self.truncations.get(agent, False):
      return mask

    player_idx = self.agent_name_mapping[agent]
    if player_idx != self.engine.current_player:
      return mask

    for move in self.engine.get_all_legal_moves():
      mask[self.encode_move(move)] = 1

    return mask

  @staticmethod
  def pos_to_index(pos: Position) -> int:
    r, c = pos
    return r * BOARD_SIZE + c

  @staticmethod
  def index_to_pos(index: int) -> Position:
    return divmod(index, BOARD_SIZE)

  @classmethod
  def encode_move(cls, move: Move) -> int:
    from_idx = cls.pos_to_index(move.start)
    to_idx = cls.pos_to_index(move.end)
    return from_idx * NUM_SQUARES + to_idx

  @classmethod
  def decode_action(cls, action: int) -> Tuple[Position, Position]:
    from_idx = action // NUM_SQUARES
    to_idx = action % NUM_SQUARES
    return cls.index_to_pos(from_idx), cls.index_to_pos(to_idx)

def env(render_mode: Optional[str] = None):
  environment =  Checkers6x6AECEnv(render_mode=render_mode)

  #  Wrappers to help prevent illegal actions, ensuring correct agent order
  # Required for compatibility with RL libraries
  environment = wrappers.CaptureStdoutWrapper(environment)
  environment = wrappers.AssertOutOfBoundsWrapper(environment)
  environment = wrappers.OrderEnforcingWrapper(environment)

  return environment

#####################
# TESTING for TASK 1 #
######################
'''
Full scale testing for Task 1
- One normal opening test
- one forced-capture test
- one multi-jump test
- one full random-play test
'''

def print_divider(title: str) -> None:
  print("\n" + "=" * 60)
  print(title)
  print("=" * 60)

def test_opening_position() -> None:
  """
  Test 1: Verify initial board setup and opening legal moves
  """
  print_divider("TEST 1: NORMAL OPENING TEST")

  environment = Checkers6x6AECEnv(render_mode="human")
  environment.reset()

  obs = environment.observe("player_0")
  mask = obs["action_mask"]
  legal_actions = np.where(mask == 1)[0]

  expected_num_actions = 5

  print("Initial observation for player_0:")
  print(obs["Observation"])
  print(f"Legal actions: {legal_actions.tolist()}")
  print(f"Number of legal actions: {len(legal_actions)}")

  assert len(legal_actions) == expected_num_actions, (
      f"Expected {expected_num_actions} legal opening actions, "
      f"but got {len(legal_actions)}"
  )

  print("PASS: Opening position legal moves are correct.")

def test_forced_capture() -> None:
  """
  Test 2: Verify mandatory capture rule.
  Only capture actions should be legal if any capture exists.
  """
  print_divider("TEST 2: FORCED CAPTURE TEST")

  environment = Checkers6x6AECEnv(render_mode="human")
  environment.reset()

  # Custom board:
  # Red at (5, 0), black at (4, 1), landing at (3, 2) empty
  environment.engine.board = [
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, -1, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, -1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0],
  ]
  environment.engine.current_player = 0
  environment.engine.winner = None
  environment.engine.forced_piece = None
  environment.agent_selection = "player_0"
  environment._update_infos()

  environment.render()

  obs = environment.observe("player_0")
  legal_actions = np.where(obs["action_mask"] == 1)[0].tolist()

  expected_move = environment.encode_move(
      environment.engine.get_all_legal_moves()[0]
  )

  print(f"Legal action: {legal_actions}")

  assert len(legal_actions) == 1, (
      f"Expected exactly 1 legal action, but got {len(legal_actions)}"
  )

  assert legal_actions[0] == expected_move, (
      f"Expected forced capture action {expected_move}, but got {legal_actions[0]}"
  )

  legal_move = environment.engine.get_all_legal_moves()[0]
  assert legal_move.is_capture, "Expected the only legal move to be a capture."

  print("PASS: Mandatory capture is enforced correctly.")

def test_multi_jump() -> None:
  """
  Test 3: Verify multi-jump continuation.
  After first capture, same player should move again with forced_piece set.
  """
  print_divider("TEST 3: MULTI-JUMP TEST")

  environment = Checkers6x6AECEnv(render_mode="human")
  environment.reset()

  # Same custom board as forced capture test, but supports second jump:
  # (5, 0) -> (3, 2) captures (4, 1)
  # then (3, 2) -> (1, 4) captures (2, 3)
  environment.engine.board = [
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, -1, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, -1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0],
  ]
  environment.engine.current_player = 0
  environment.engine.winner = None
  environment.engine.forced_piece = None
  environment.agent_selection = "player_0"
  environment._update_infos()

  print("Before first jump:")
  environment.render()

  obs = environment.observe("player_0")
  legal_actions = np.where(obs["action_mask"] == 1)[0]
  assert len(legal_actions) == 1, "Expected exactly one first capture action."

  first_action = int(legal_actions[0])
  environment.step(first_action)

  print("After first jump: ")
  environment.render()

  # Check same player still moves
  assert environment.agent_selection == "player_0", (
      "Expected player_0 to continue turn after first jump."
  )
  assert environment.engine.forced_piece == (3, 2), (
      f"Expected forced_piece to be (3, 2), got {environment.engine.forced_piece}"
  )

  obs2 = environment.observe("player_0") # Corrected typo 'plyaer_0' to 'player_0'
  legal_actions2 = np.where(obs2["action_mask"] == 1)[0].tolist()

  print(f"Legal actions after first jump: {legal_actions2}")

  assert len(legal_actions2) == 1, (
      f"Expected exactly 1 continuation jump, got {len(legal_actions2)}"
  )

  second_move = environment.engine.get_all_legal_moves()[0]
  assert second_move.is_capture, "Expected continuation move to also be a capture."

  # Apply second jump
  second_action = legal_actions2[0]
  environment.step(second_action)

  print("After second jump: ")
  environment.render()

  # After second jump, either:
  # 1. the turn switches to player_1 if game continues, or
  # 2. the game terminates immediately if player_1 has no pieces / no moves

  if any(environment.terminations.values()):
      assert environment.engine.winner == 0, (
          f"Expected player_0 to win, got winner={environment.engine.winner}"
      )
      print("Game terminated correctly after second jump. player_0 wins.")
  else:
      assert environment.agent_selection == "player_1", (
          "Expected turn to switch to player_1 after jump chain ends."
      )
      assert environment.engine.forced_piece is None, (
          "Expected forced_piece to be cleared after jump chain ends."
      )
      print("Turn switched correctly after second jump.")

  print("PASS: Multi-jump logic works correctly.")

def test_full_random_play(max_cycles: int = 200) -> None:
  """
  Test 4: Run a full random game using only legal actions.
  Confirms env stability, turn handling, and termination.
  """
  print_divider("TEST 4: FULL RANDOM-PLAY TEST")

  environment = env(render_mode="human")
  environment.reset()

  step_count = 0
  final_rewards = None

  for agent in environment.agent_iter(max_iter=max_cycles):
    obs, reward, termination, truncation, info = environment.last()

    print(f"\nAgent: {agent}")
    print(f"Reward: {reward}")
    if termination or truncation:
      action = None
    else:
      legal_actions = np.where(obs["action_mask"] == 1)[0]
      print(f"Num legal actions; {len(legal_actions)}")

      if len(legal_actions) > 0:
        action = int(np.random.choice(legal_actions))
      else:
        action = None

    environment.step(action)
    step_count += 1

    if all(environment.terminations.values()) or all(environment.truncations.values()):
      final_rewards = dict(environment.rewards)
      break

  print("\nFinal rewards:", final_rewards)
  print("Terminations: ", environment.terminations)
  print("Truncations: ", environment.truncations)
  print("Total steps: ", step_count)

  assert final_rewards is not None, "Expected game to terminate or truncate."
  assert (
      all(environment.terminations.values()) or all(environment.truncations.values())
  ), "Expected game to end in termination or truncation."

  print("PASS: Full random-play test completed successfully.")

if __name__ == "__main__":
  test_opening_position()
  test_forced_capture()
  test_multi_jump()
  test_full_random_play()
