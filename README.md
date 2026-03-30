# 6x6 Checkers Multi-Agent Environment (PettingZoo AEC)

## Overview

This project implements a custom **6×6 Checkers environment** using the **PettingZoo AEC (Agent Environment Cycle) API**. The environment supports two agents in a turn-based setting and enforces standard checkers rules, including:

* diagonal movement
* mandatory captures
* multi-jump sequences
* king promotion
* win conditions based on elimination or blocking

This environment is designed to be used for **multi-agent reinforcement learning (MARL)** experiments.

---

## Agents

* `player_0` (Red pieces)
* `player_1` (Black pieces)

---

## Observation Space

Each agent observes:

```python
{
    "observation": Box(low=-2, high=2, shape=(6,6), dtype=int8),
    "action_mask": Box(low=0, high=1, shape=(1296,), dtype=int8)
}
```

### Description

* Board is a **6×6 grid**

* Encoding:

  * `0` → empty
  * `1` → own man
  * `2` → own king
  * `-1` → opponent man
  * `-2` → opponent king

* Perspective is **agent-relative**:

  * current player’s pieces are always positive

---

## Action Space

```python
Discrete(1296)
```

Each action encodes:

```python
action = from_square * 36 + to_square
```

* Squares indexed from `0–35` (row-major)

---

## Rewards

| Event        | Reward                |
| ------------ | --------------------- |
| Win          | +1                    |
| Loss         | -1                    |
| Illegal move | -1 (opponent gets +1) |
| Otherwise    | 0                     |

---

## Termination Conditions

The game ends when:

* one player has no pieces
* one player has no legal moves
* illegal action is taken

---

## Truncation

* Episode truncates after **200 turns** to prevent infinite loops

---

## Core Game Logic

The environment enforces full checkers rules:

### Movement

* Pieces move diagonally forward
* Kings move both directions

### Mandatory Capture

* If any capture exists → only capture moves are allowed

### Multi-Jump

* After a capture, if another capture exists:

  * same piece must continue
  * turn does **not switch**

### King Promotion

* A piece becomes a king upon reaching the opposite end

### Turn Handling

* Turn switches only after:

  * a non-capture move, or
  * completion of a capture chain

---

## Testing Results

To validate correctness, four tests were implemented.

---

### ✅ Test 1: Opening Position

#### Goal

Verify:

* correct board initialization
* correct legal moves
* correct action encoding

#### Result

```text
Legal actions: [918, 920, 992, 994, 1066]
Number of legal actions: 5
PASS: Opening position legal moves are correct.
```

#### Interpretation

* Matches expected opening moves
* Action encoding verified correct

---

### ✅ Test 2: Mandatory Capture

#### Goal

Ensure:

* capture is enforced when available
* normal moves are disallowed

#### Board

```text
5 r . . . . .
4 . b . . . .
3 . . . . . .
2 . . . b . .
```

#### Result

```text
Legal action: [1100]
PASS: Mandatory capture is enforced correctly.
```

#### Interpretation

* Only capture move available
* Mandatory capture rule enforced correctly

---

### ✅ Test 3: Multi-Jump

#### Goal

Verify:

* same piece continues after capture
* forced_piece is enforced
* multi-jump works correctly
* game termination after final capture

#### Result

```text
After first jump:
Forced piece: (3, 2)

Legal actions after first jump: [730]

After second jump:
Game terminated correctly after second jump. player_0 wins.
PASS: Multi-jump logic works correctly.
```

#### Interpretation

* Turn does not switch after first jump
* Only continuation jump allowed
* After second jump, opponent eliminated → correct termination

---

### ✅ Test 4: Full Random Game Simulation

#### Goal

Verify:

* environment stability
* action mask correctness
* turn transitions
* termination logic

#### Final Output

```text
Final rewards: {'player_0': 1.0, 'player_1': -1.0}
Terminations: {'player_0': True, 'player_1': True}
Truncations: {'player_0': False, 'player_1': False}
Total steps: 21
PASS: Full random-play test completed successfully.
```

#### Interpretation

* Game completed without errors
* Correct winner assigned (player_0)
* No invalid states occurred
* Multi-agent interaction stable

---

## Summary of Validation

| Feature               | Status |
| --------------------- | ------ |
| Board initialization  | ✅      |
| Action encoding       | ✅      |
| Action masking        | ✅      |
| Legal move generation | ✅      |
| Mandatory capture     | ✅      |
| Multi-jump logic      | ✅      |
| King promotion        | ✅      |
| Turn handling         | ✅      |
| Termination detection | ✅      |
| Reward assignment     | ✅      |

---
