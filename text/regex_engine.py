r"""
游戏内正则表达式引擎与NFA/DFA系统 - 核心模块

支持:
  - 元字符: . * + ? | ()
  - 字符类: [a-z] [^abc] \d \D \w \W \s \S
  - 锚点与转义: ^ $ \n \t 等
  - Thompson 构造 NFA (epsilon 转换 + 分组)
  - 带回溯剪枝与缓存的 NFA 模拟 (捕获组编号/嵌套)
  - 子集构造 NFA -> DFA, 并做 Hopcroft 最小化
纯 Python 标准库实现, 正确处理 Unicode (玩家名等中文文本).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple
import re as _re


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

class Node:
    pass


@dataclass
class Char(Node):
    # kind: 'char' | 'any' | 'class' | 'anchor'
    kind: str
    ch: Optional[str] = None
    chars: Optional[FrozenSet[str]] = None          # 类中的显式字符
    ranges: Optional[Tuple[Tuple[str, str], ...]] = None
    digit: Optional[bool] = None                    # True 含 \d, False 含 \D
    word: Optional[bool] = None
    space: Optional[bool] = None
    negated: bool = False


@dataclass
class Concat(Node):
    children: List[Node]


@dataclass
class Alt(Node):
    children: List[Node]


@dataclass
class Repeat(Node):
    child: Node
    min: int
    max: Optional[int]                              # None 表示无限


@dataclass
class Group(Node):
    child: Node
    index: int
    name: Optional[str] = None
    capturing: bool = True


@dataclass
class Lazy(Node):
    child: Node


# ---------------------------------------------------------------------------
# 模式解析 (递归下降)
# ---------------------------------------------------------------------------

_ESCAPES = {
    'n': '\n', 't': '\t', 'r': '\r', 'f': '\f',
    'v': '\v', '0': '\0', 'a': '\a', 'b': '\b',
}

# 简写类 -> (digit, word, space) 标志
_SHORTHAND = {
    'd': (True, None, None),
    'D': (False, None, None),
    'w': (None, True, None),
    'W': (None, False, None),
    's': (None, None, True),
    'S': (None, None, False),
}


class PatternSyntaxError(Exception):
    pass


class _Parser:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0
        self.group_count = 0
        self.names: Dict[str, int] = {}

    def error(self, msg: str):
        raise PatternSyntaxError(f"{msg} (position {self.pos}): {self.pattern!r}")

    def peek(self) -> str:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else ''

    def parse(self) -> Node:
        node = self.parse_alternation()
        if self.pos != len(self.pattern):
            self.error("unexpected ')'")
        return node

    # alternation = concat ('|' concat)*
    def parse_alternation(self) -> Node:
        children = [self.parse_concat()]
        while self.peek() == '|':
            self.pos += 1
            children.append(self.parse_concat())
        return children[0] if len(children) == 1 else Alt(children)

    # concat = quantified*  (直到 ')' 或 '|' 或结尾)
    def parse_concat(self) -> Node:
        children: List[Node] = []
        while self.peek() and self.peek() not in ')|':
            children.append(self.parse_quantified())
        if not children:
            return Concat([])
        return children[0] if len(children) == 1 else Concat(children)

    def parse_quantified(self) -> Node:
        atom = self.parse_atom()
        c = self.peek()
        if c == '*':
            self.pos += 1
            atom = Repeat(atom, 0, None)
        elif c == '+':
            self.pos += 1
            atom = Repeat(atom, 1, None)
        elif c == '?':
            self.pos += 1
            atom = Repeat(atom, 0, 1)
        elif c == '{':
            start = self.pos + 1
            end = self.pattern.find('}', start)
            if end != -1:
                spec = self.pattern[start:end]
                m = _re.fullmatch(r'(\d*)(,?)(\d*)', spec)
                if m and (m.group(1) or m.group(3)):
                    lo = int(m.group(1)) if m.group(1) else 0
                    if m.group(2):
                        hi = int(m.group(3)) if m.group(3) else None
                    else:
                        hi = lo
                    if hi is not None and hi < lo:
                        self.error("invalid quantifier range")
                    self.pos = end + 1
                    atom = Repeat(atom, lo, hi)
        # 惰性量词 *? +? ?? {n,m}?
        if self.peek() == '?':
            self.pos += 1
            atom = Lazy(atom)
        return atom

    def parse_atom(self) -> Node:
        c = self.peek()
        if c == '(':
            return self.parse_group()
        if c == '[':
            return self.parse_class()
        if c == '.':
            self.pos += 1
            return Char('any')
        if c == '^':
            self.pos += 1
            return Char('anchor', ch='^')
        if c == '$':
            self.pos += 1
            return Char('anchor', ch='$')
        if c == '\\':
            return self.parse_escape()
        if c in '*+?':
            self.error(f"unexpected quantifier '{c}'")
        self.pos += 1
        return Char('char', ch=c)

    def parse_group(self) -> Node:
        assert self.peek() == '('
        self.pos += 1
        name: Optional[str] = None
        capturing = True
        if self.peek() == '?':
            self.pos += 1
            nc = self.peek()
            if nc == ':':
                self.pos += 1
                capturing = False
            elif nc == 'P':
                self.pos += 1
                if self.peek() == '<':
                    self.pos += 1
                    end = self.pattern.find('>', self.pos)
                    if end == -1:
                        self.error("named group missing '>'")
                    name = self.pattern[self.pos:end]
                    if not name or not _re.fullmatch(r'\w+', name, _re.UNICODE):
                        self.error("invalid group name")
                    self.pos = end + 1
                else:
                    self.error("unsupported group syntax")
            elif nc == '<':
                self.pos += 1
                end = self.pattern.find('>', self.pos)
                if end == -1:
                    self.error("named group missing '>'")
                name = self.pattern[self.pos:end]
                if not name or not _re.fullmatch(r'\w+', name, _re.UNICODE):
                    self.error("invalid group name")
                self.pos = end + 1
            else:
                self.error("unsupported group syntax")
        if capturing:
            self.group_count += 1
            index = self.group_count
            if name is not None:
                if name in self.names:
                    self.error(f"duplicate group name {name!r}")
                self.names[name] = index
        else:
            index = 0
        child = self.parse_alternation()
        if self.peek() != ')':
            self.error("group missing ')'")
        self.pos += 1
        if not capturing:
            return child
        return Group(child, index, name, True)

    def parse_class(self) -> Node:
        assert self.peek() == '['
        self.pos += 1
        negated = False
        if self.peek() == '^':
            negated = True
            self.pos += 1
        chars: set = set()
        ranges: List[Tuple[str, str]] = []
        digit = word = space = None
        first = True
        while first or self.peek() != ']':
            first = False
            if not self.peek():
                self.error("character class missing ']'")
            atom = self._read_class_atom()
            if isinstance(atom, tuple):
                # 简写类标记
                key = atom[1]
                d, w, s = _SHORTHAND[key]
                digit = self._merge_flag(digit, d)
                word = self._merge_flag(word, w)
                space = self._merge_flag(space, s)
                continue
            if (self.peek() == '-' and self.pos + 1 < len(self.pattern)
                    and self.pattern[self.pos + 1] != ']'):
                self.pos += 1  # '-'
                atom2 = self._read_class_atom()
                if isinstance(atom2, tuple):
                    self.error("range endpoint cannot be a shorthand class")
                if ord(atom) > ord(atom2):
                    self.error("reversed character range")
                ranges.append((atom, atom2))
            else:
                chars.add(atom)
        self.pos += 1  # ']'
        return Char('class', chars=frozenset(chars), ranges=tuple(ranges),
                    digit=digit, word=word, space=space, negated=negated)

    @staticmethod
    def _merge_flag(cur, val):
        if val is None:
            return cur
        if cur is None:
            return val
        if cur != val:
            return 'both'
        return cur

    def _read_class_atom(self):
        c = self.peek()
        if c == '\\':
            esc = self.parse_escape()
            if esc.kind == 'class':
                # 简写类
                if esc.digit is not None:
                    return ('short', 'd' if esc.digit else 'D')
                if esc.word is not None:
                    return ('short', 'w' if esc.word else 'W')
                return ('short', 's' if esc.space else 'S')
            return esc.ch
        if c == '':
            self.error("unterminated character class")
        self.pos += 1
        return c

    def parse_escape(self) -> Node:
        assert self.peek() == '\\'
        self.pos += 1
        c = self.peek()
        if c == '':
            self.error("trailing backslash")
        if c in _SHORTHAND:
            self.pos += 1
            d, w, s = _SHORTHAND[c]
            return Char('class', chars=frozenset(), ranges=(),
                        digit=d, word=w, space=s)
        if c in _ESCAPES:
            self.pos += 1
            return Char('char', ch=_ESCAPES[c])
        if c == 'x':
            hexd = self.pattern[self.pos + 1:self.pos + 3]
            if len(hexd) == 2 and _re.fullmatch(r'[0-9a-fA-F]{2}', hexd):
                self.pos += 3
                return Char('char', ch=chr(int(hexd, 16)))
            self.error("\\x requires two hex digits")
        if c == 'u':
            hexd = self.pattern[self.pos + 1:self.pos + 5]
            if len(hexd) == 4 and _re.fullmatch(r'[0-9a-fA-F]{4}', hexd):
                self.pos += 5
                return Char('char', ch=chr(int(hexd, 16)))
            self.error("\\u requires four hex digits")
        # 其它转义: 转义后的字面量 (元字符/标点)
        self.pos += 1
        return Char('char', ch=c)


# ---------------------------------------------------------------------------
# 字符匹配谓词
# ---------------------------------------------------------------------------

def _is_digit(ch: str) -> bool:
    return ch.isdigit()


def _is_word(ch: str) -> bool:
    return ch == '_' or (ch.isalnum())


def _is_space(ch: str) -> bool:
    return ch in ' \t\n\r\f\v' or ch.isspace()


def _char_matches(node: Char, ch: str) -> bool:
    if node.kind == 'char':
        return ch == node.ch
    if node.kind == 'any':
        return ch != '\n'
    if node.kind == 'anchor':
        return False
    # class
    matched = False
    if ch in (node.chars or ()):
        matched = True
    if not matched:
        for lo, hi in node.ranges or ():
            if lo <= ch <= hi:
                matched = True
                break
    if not matched:
        if node.digit is True and _is_digit(ch):
            matched = True
        if node.word is True and _is_word(ch):
            matched = True
        if node.space is True and _is_space(ch):
            matched = True
    if node.negated:
        return not matched
    # 正向类中混入 \D \W \S
    extra = False
    if node.digit is False and not _is_digit(ch):
        extra = True
    if node.word is False and not _is_word(ch):
        extra = True
    if node.space is False and not _is_space(ch):
        extra = True
    return matched or extra


# ---------------------------------------------------------------------------
# NFA (Thompson 构造, epsilon 转换 + 分组动作)
# ---------------------------------------------------------------------------

@dataclass
class State:
    id: int
    epsilon: List[Tuple[int, Optional[Tuple[str, int]]]] = field(default_factory=list)
    # label: None 表示通配任意单字符(供占位), 这里统一放 Char 节点
    transitions: List[Tuple[Char, int]] = field(default_factory=list)
    accept: bool = False


@dataclass
class NFA:
    states: List[State]
    start: int
    accept: int
    groups: Dict[int, Optional[str]]


class _Builder:
    def __init__(self):
        self.states: List[State] = []
        self.groups: Dict[int, Optional[str]] = {}

    def new_state(self) -> int:
        sid = len(self.states)
        self.states.append(State(sid))
        return sid

    def link(self, src: int, dst: int, action: Optional[Tuple[str, int]] = None):
        self.states[src].epsilon.append((dst, action))

    def build(self, node: Node) -> Tuple[int, int]:
        t = type(node)
        if t is Char:
            if node.kind == 'anchor':
                s = self.new_state()
                a = self.new_state()
                self.states[s].transitions.append((node, a))
                return s, a
            s = self.new_state()
            a = self.new_state()
            self.states[s].transitions.append((node, a))
            return s, a
        if t is Concat:
            if not node.children:
                s = self.new_state()
                return s, s
            first_s, last_a = self.build(node.children[0])
            for child in node.children[1:]:
                cs, ca = self.build(child)
                self.link(last_a, cs)
                last_a = ca
            return first_s, last_a
        if t is Alt:
            s = self.new_state()
            a = self.new_state()
            for child in node.children:
                cs, ca = self.build(child)
                self.link(s, cs)
                self.link(ca, a)
            return s, a
        if t is Group:
            self.groups[node.index] = node.name
            cs, ca = self.build(node.child)
            s = self.new_state()
            a = self.new_state()
            self.link(s, cs, ('open', node.index))
            self.link(ca, a, ('close', node.index))
            return s, a
        if t in (Repeat, Lazy):
            lazy = t is Lazy
            inner = node.child if lazy else node
            lo, hi, body = inner.min, inner.max, inner.child

            def make_body():
                return self.build(body)

            s = self.new_state()
            cur = s
            for _ in range(lo):
                bs, ba = make_body()
                self.link(cur, bs)
                cur = ba
            a = self.new_state()
            if hi is None:
                bs, ba = make_body()
                # 贪婪: 进入循环体优先于退出; 惰性: 退出优先
                if lazy:
                    self.link(cur, a)
                    self.link(cur, bs)
                    self.link(ba, a)
                    self.link(ba, bs)
                else:
                    self.link(cur, bs)
                    self.link(cur, a)
                    self.link(ba, bs)
                    self.link(ba, a)
            else:
                extra = hi - lo
                for _ in range(extra):
                    bs, ba = make_body()
                    if lazy:
                        # 惰性: 退出边优先于进入可选副本
                        self.link(cur, a)
                        self.link(cur, bs)
                    else:
                        self.link(cur, bs)
                        self.link(cur, a)
                    cur = ba
                self.link(cur, a)
            return s, a
        raise PatternSyntaxError(f"unsupported AST node: {node!r}")


def _closure(nfa: NFA, items, pos: int):
    """epsilon 闭包 (按边优先级 DFS, 先到先得剪枝).
    items: [(state_id, caps)]. 返回 [(state_id, caps)] 优先级有序列表,
    每个状态携带到达它时的捕获组快照."""
    order = []
    seen = set()

    def visit(sid: int, caps):
        if sid in seen:
            return
        seen.add(sid)
        order.append((sid, caps))
        for dst, action in nfa.states[sid].epsilon:
            if action is not None:
                kind, gidx = action
                slot = 2 * gidx if kind == 'open' else 2 * gidx + 1
                caps2 = list(caps)
                caps2[slot] = pos
                visit(dst, tuple(caps2))
            else:
                visit(dst, caps)

    for sid, caps in items:
        visit(sid, caps)
    return order


class Match:
    """匹配结果: 提供 group()/groups()/span() 等接口."""

    def __init__(self, text: str, start: int, end: int,
                 caps: Tuple, group_names: Dict[str, int]):
        self.string = text
        self._start = start
        self._end = end
        self._caps = caps
        self._names = group_names

    def group(self, *args):
        if not args:
            return self.string[self._start:self._end]
        if len(args) == 1:
            return self._group_one(args[0])
        return tuple(self._group_one(a) for a in args)

    def _group_one(self, key):
        if isinstance(key, str):
            if key not in self._names:
                raise IndexError(f"unknown group name {key!r}")
            key = self._names[key]
        if key == 0:
            return self.string[self._start:self._end]
        lo = self._caps[2 * key]
        hi = self._caps[2 * key + 1]
        if lo is None or hi is None:
            return None
        return self.string[lo:hi]

    def groups(self, default=None):
        n = (len(self._caps) - 2) // 2
        out = []
        for i in range(1, n + 1):
            g = self._group_one(i)
            out.append(default if g is None else g)
        return tuple(out)

    def groupdict(self, default=None):
        return {name: (self._group_one(idx) if self._group_one(idx) is not None else default)
                for name, idx in self._names.items()}

    def span(self, group=0):
        if group == 0:
            return (self._start, self._end)
        if isinstance(group, str):
            group = self._names[group]
        return (self._caps[2 * group], self._caps[2 * group + 1])

    def start(self, group=0):
        return self.span(group)[0]

    def end(self, group=0):
        return self.span(group)[1]

    @property
    def pos(self):
        return self._start

    @property
    def endpos(self):
        return self._end

    def __bool__(self):
        return True

    def __repr__(self):
        return f"<Match {self.string[self._start:self._end]!r} span=({self._start}, {self._end})>"


def _simulate(nfa: NFA, text: str, start_pos: int,
              group_names: Dict[str, int],
              forbid_empty: bool = False) -> Optional[Match]:
    """从 start_pos 起模拟 NFA.
    按优先级顺序逐状态尝试 (贪婪/惰性由 epsilon 边顺序决定),
    剪枝+缓存: 记忆化 (位置, 活跃状态集), 已失败的组合不再重复探索."""
    n_groups = len(nfa.groups)
    caps0 = [None] * (2 * (n_groups + 1))
    caps0[0] = start_pos
    memo = set()  # (pos, frozenset(活跃状态)) 已失败 -> 剪枝

    def step(items, pos):
        order = _closure(nfa, items, pos)
        key = (pos, frozenset(s for s, _ in order))
        if key in memo:
            return None
        ch = text[pos] if pos < len(text) else None
        for sid, caps in order:
            # forbid_empty 时禁止在起始位置直接接受 (用于零宽匹配后的推进)
            if sid == nfa.accept and not (forbid_empty and pos == start_pos):
                return caps, pos
            for label, dst in nfa.states[sid].transitions:
                if label.kind == 'anchor':
                    ok = ((label.ch == '^' and pos == 0) or
                          (label.ch == '$' and pos == len(text)))
                    if ok:
                        r = step([(dst, caps)], pos)
                        if r is not None:
                            return r
                elif ch is not None and _char_matches(label, ch):
                    r = step([(dst, caps)], pos + 1)
                    if r is not None:
                        return r
        memo.add(key)
        return None

    result = step([(nfa.start, tuple(caps0))], start_pos)
    if result is None:
        return None
    caps, end_pos = result
    caps = list(caps)
    caps[0] = start_pos
    caps[1] = end_pos
    return Match(text, start_pos, end_pos, tuple(caps), group_names)


# ---------------------------------------------------------------------------
# DFA (子集构造 + Hopcroft 最小化)
# ---------------------------------------------------------------------------

def _epsilon_closure_set(nfa: NFA, ids) -> FrozenSet[int]:
    seen = set(ids)
    stack = list(seen)
    while stack:
        sid = stack.pop()
        for dst, _action in nfa.states[sid].epsilon:
            if dst not in seen:
                seen.add(dst)
                stack.append(dst)
        # 锚点转换在 DFA 中按"可满足"处理: 并入闭包会改变语义,
        # 因此 DFA 仅用于不含锚点的模式; 含锚点走 NFA 模拟.
        for label, dst in nfa.states[sid].transitions:
            if label.kind == 'anchor':
                raise ValueError('anchor not supported in DFA')
    return frozenset(seen)


def _move(nfa: NFA, ids: FrozenSet[int], ch: str) -> FrozenSet[int]:
    out = set()
    for sid in ids:
        for label, dst in nfa.states[sid].transitions:
            if label.kind != 'anchor' and _char_matches(label, ch):
                out.add(dst)
    return frozenset(out)


class DFA:
    """确定性有限自动机: 由 NFA 子集构造而来, 并做状态最小化."""

    def __init__(self, nfa: NFA):
        self.nfa = nfa
        # 子集构造 (惰性, 按字符即时展开, 缓存结果)
        self._subset_cache: Dict[FrozenSet[int], int] = {}
        self._states: List[FrozenSet[int]] = []
        self._trans: Dict[Tuple[int, str], int] = {}
        self.accepting: set = set()
        self.start = self._subset_id(_epsilon_closure_set(nfa, {nfa.start}))
        # 最小化结果 (惰性计算)
        self._minimized = False
        self.min_states: List[List[int]] = []       # 等价类 -> 原状态id列表
        self._state_to_class: Dict[int, int] = {}
        self.min_start: int = 0
        self.min_accepting: set = set()

    def _subset_id(self, subset: FrozenSet[int]) -> int:
        if subset not in self._subset_cache:
            self._subset_cache[subset] = len(self._states)
            self._states.append(subset)
            if self.nfa.accept in subset:
                self.accepting.add(self._subset_cache[subset])
        return self._subset_cache[subset]

    def step(self, state: int, ch: str) -> Optional[int]:
        key = (state, ch)
        if key in self._trans:
            return self._trans[key]
        moved = _move(self.nfa, self._states[state], ch)
        if not moved:
            self._trans[key] = None
            return None
        nxt = self._subset_id(_epsilon_closure_set(self.nfa, moved))
        self._trans[key] = nxt
        return nxt

    def run(self, text: str, pos: int = 0) -> Optional[int]:
        """从 pos 起贪婪运行, 返回最长匹配终点; 无匹配返回 None."""
        state = self.start
        last_accept = pos if state in self.accepting else None
        i = pos
        while i < len(text):
            nxt = self.step(state, text[i])
            if nxt is None:
                break
            state = nxt
            i += 1
            if state in self.accepting:
                last_accept = i
        return last_accept

    # -- 最小化 (Hopcroft) ------------------------------------------------
    def _explore_all(self):
        """用转移标签的代表字符探测出所有可达子集状态."""
        # 收集 NFA 中出现的所有谓词的代表字符
        reps: List[str] = []
        for st in self.nfa.states:
            for label, _ in st.transitions:
                if label.kind == 'anchor':
                    continue
                reps.extend(_representative_chars(label))
        queue = [self.start]
        seen = {self.start}
        while queue:
            cur = queue.pop()
            for ch in reps:
                nxt = self.step(cur, ch)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return sorted(seen)

    def minimize(self):
        if self._minimized:
            return self
        states = self._explore_all()
        n = len(self._states)
        # 初始划分: 接受 / 非接受
        accept = sorted(s for s in states if s in self.accepting)
        non_accept = sorted(s for s in states if s not in self.accepting)
        parts: List[set] = [p for p in (set(accept), set(non_accept)) if p]
        # 签名细分: 按 (是否接受, 各代表字符的目标类) 迭代细化
        reps: List[str] = []
        for st in self.nfa.states:
            for label, _ in st.transitions:
                if label.kind != 'anchor':
                    reps.extend(_representative_chars(label))
        changed = True
        while changed:
            changed = False
            cls = {}
            for i, p in enumerate(parts):
                for s in p:
                    cls[s] = i
            new_parts: List[set] = []
            for p in parts:
                buckets: Dict[Tuple, set] = {}
                for s in p:
                    sig = []
                    for ch in reps:
                        t = self.step(s, ch)
                        sig.append(None if t is None else cls.get(t, -1))
                    buckets.setdefault(tuple(sig), set()).add(s)
                if len(buckets) > 1:
                    changed = True
                new_parts.extend(buckets.values())
            parts = new_parts
        self.min_states = [sorted(p) for p in parts]
        self._state_to_class = {}
        for i, p in enumerate(self.min_states):
            for s in p:
                self._state_to_class[s] = i
        self.min_start = self._state_to_class[self.start]
        self.min_accepting = {self._state_to_class[s] for s in accept}
        self._minimized = True
        return self

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def minimized_state_count(self) -> int:
        self.minimize()
        return len(self.min_states)


def _representative_chars(label: Char) -> List[str]:
    """为谓词挑选代表字符, 用于 DFA 状态探测与最小化."""
    reps: List[str] = []
    if label.kind == 'char':
        return [label.ch]
    if label.kind == 'any':
        return ['a', '0', '_', '张']
    # class
    chars = set(label.chars or ())
    for lo, hi in label.ranges or ():
        chars.add(lo)
        chars.add(hi)
        if ord(lo) + 1 < ord(hi):
            chars.add(chr((ord(lo) + ord(hi)) // 2))
    if label.digit is True:
        chars.add('5')
    if label.word is True:
        chars.add('a')
        chars.add('_')
        chars.add('张')
    if label.space is True:
        chars.add(' ')
    if label.digit is False or label.word is False or label.space is False or label.negated:
        # 反向集合: 用探测字符覆盖正反两面
        chars.update(['a', 'Z', '5', ' ', '_', '-', '张', '\t', '.'])
    return sorted(chars)


# ---------------------------------------------------------------------------
# RegexEngine 主类
# ---------------------------------------------------------------------------

@dataclass
class Config:
    pass


class _Compiled:
    __slots__ = ('pattern', 'ast', 'group_count', 'group_names', 'nfa', 'dfa')

    def __init__(self, pattern, ast, group_count, group_names):
        self.pattern = pattern
        self.ast = ast
        self.group_count = group_count
        self.group_names = group_names
        self.nfa: Optional[NFA] = None
        self.dfa: Optional[DFA] = None


class RegexEngine:
    """游戏内正则表达式引擎: 解析 -> NFA -> 模拟匹配 / DFA -> 捕获组."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._cache: Dict[str, _Compiled] = {}

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._cache = {}

    # -- 编译管线 ---------------------------------------------------------
    def parse_pattern(self, pattern: str) -> Node:
        """解析模式字符串为 AST, 支持元字符 . * + ? | () 与字符类."""
        parser = _Parser(pattern)
        ast = parser.parse()
        compiled = _Compiled(pattern, ast, parser.group_count,
                             {v: k for k, v in parser.names.items()})
        # group_names 存 name->index 更实用
        compiled.group_names = dict(parser.names)
        self._cache[pattern] = compiled
        return ast

    def _compile(self, pattern: str) -> _Compiled:
        if pattern in self._cache:
            return self._cache[pattern]
        self.parse_pattern(pattern)
        return self._cache[pattern]

    def build_nfa(self, pattern) -> NFA:
        """由模式(或 AST)构建 Thompson NFA, 支持 epsilon 转换与分组."""
        if isinstance(pattern, Node):
            # 重新统计分组数
            count = self._count_groups(pattern)
            ast = pattern
            group_count = count
        else:
            compiled = self._compile(pattern)
            if compiled.nfa is not None:
                return compiled.nfa
            ast = compiled.ast
            group_count = compiled.group_count
        builder = _Builder()
        start, accept = builder.build(ast)
        builder.states[accept].accept = True
        nfa = NFA(builder.states, start, accept, builder.groups)
        if not isinstance(pattern, Node):
            compiled.nfa = nfa
        return nfa

    @staticmethod
    def _count_groups(node: Node) -> int:
        best = 0
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, Group):
                best = max(best, n.index)
                stack.append(n.child)
            elif isinstance(n, (Concat, Alt)):
                stack.extend(n.children)
            elif isinstance(n, Repeat):
                stack.append(n.child)
            elif isinstance(n, Lazy):
                stack.append(n.child)
        return best

    def build_dfa(self, pattern) -> DFA:
        """由 NFA 子集构造 DFA (含锚点的模式回退 NFA 模拟)."""
        compiled = self._compile(pattern) if isinstance(pattern, str) else None
        if compiled is not None and compiled.dfa is not None:
            return compiled.dfa
        nfa = self.build_nfa(pattern)
        dfa = DFA(nfa)
        if compiled is not None:
            compiled.dfa = dfa
        return dfa

    # -- 匹配 -------------------------------------------------------------
    @staticmethod
    def _has_anchor(ast: Node) -> bool:
        stack = [ast]
        while stack:
            n = stack.pop()
            if isinstance(n, Char) and n.kind == 'anchor':
                return True
            if isinstance(n, (Concat, Alt)):
                stack.extend(n.children)
            elif isinstance(n, (Group, Repeat, Lazy)):
                stack.append(n.child)
        return False

    @staticmethod
    def _has_lazy(ast: Node) -> bool:
        stack = [ast]
        while stack:
            n = stack.pop()
            if isinstance(n, Lazy):
                return True
            if isinstance(n, Repeat):
                stack.append(n.child)
            elif isinstance(n, (Concat, Alt)):
                stack.extend(n.children)
            elif isinstance(n, Group):
                stack.append(n.child)
        return False

    def nfa_match(self, pattern, text: str, pos: int = 0,
                  forbid_empty: bool = False) -> Optional[Match]:
        """NFA 模拟匹配 (带回溯剪枝与缓存), 从 pos 起搜索首个匹配.

        forbid_empty=True 时, 拒绝在 pos 处的零宽匹配 (findall 扫描推进用).
        """
        compiled = self._compile(pattern) if isinstance(pattern, str) else None
        ast = compiled.ast if compiled else pattern
        group_names = compiled.group_names if compiled else {}
        nfa = self.build_nfa(pattern)
        start = pos
        while start <= len(text):
            m = _simulate(nfa, text, start, group_names,
                          forbid_empty=forbid_empty and start == pos)
            if m is not None:
                return m
            start += 1
        return None

    def dfa_match(self, pattern: str, text: str, pos: int = 0) -> Optional[Match]:
        """DFA 匹配 (无捕获组, 速度快); 含锚点/分组/惰性量词语义时回退 NFA."""
        compiled = self._compile(pattern)
        if (self._has_anchor(compiled.ast) or compiled.group_count
                or self._has_lazy(compiled.ast)):
            return self.nfa_match(pattern, text, pos)
        dfa = self.build_dfa(pattern)
        start = pos
        while start <= len(text):
            end = dfa.run(text, start)
            if end is not None:
                caps = (start, end) + (None,) * (2 * compiled.group_count)
                return Match(text, start, end, caps, compiled.group_names)
            start += 1
        return None

    def search(self, pattern: str, text: str, pos: int = 0) -> Optional[Match]:
        return self.nfa_match(pattern, text, pos)

    def match(self, pattern: str, text: str) -> Optional[Match]:
        m = self.nfa_match(pattern, text, 0)
        if m is not None and m.start() == 0:
            return m
        return None

    def fullmatch(self, pattern: str, text: str) -> Optional[Match]:
        m = self.nfa_match(pattern, text, 0)
        if m is not None and m.start() == 0 and m.end() == len(text):
            return m
        return None

    def findall(self, pattern: str, text: str) -> List:
        out = []
        pos = 0
        compiled = self._compile(pattern)
        forbid_empty = False
        while pos <= len(text):
            m = self.nfa_match(pattern, text, pos, forbid_empty=forbid_empty)
            if m is None:
                break
            if compiled.group_count == 0:
                out.append(m.group(0))
            elif compiled.group_count == 1:
                # re.findall 单组模式: 未参与的组返回空串
                out.append(m.group(1) or '')
            else:
                out.append(m.groups(default=''))
            if m.end() == m.start():
                # 零宽匹配: 留在原位, 但下一次拒绝同位零宽匹配 (与 re 行为一致)
                pos = m.end()
                forbid_empty = True
            else:
                pos = m.end()
                forbid_empty = False
        return out

    # -- 捕获组 -----------------------------------------------------------
    def capture_groups(self, pattern: str, text: str) -> Optional[Tuple]:
        """返回匹配中各捕获组的内容 (按组编号, 支持嵌套分组).
        无匹配返回 None; 无分组时返回空元组."""
        m = self.nfa_match(pattern, text)
        if m is None:
            return None
        return m.groups()

    def group_count(self, pattern: str) -> int:
        return self._compile(pattern).group_count
