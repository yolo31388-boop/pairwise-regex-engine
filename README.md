# 游戏内正则表达式引擎与NFA/DFA系统

## 项目描述

游戏内正则表达式引擎与NFA/DFA系统的bug修复项目。核心模块 `text/regex_engine.py` 存在多个已知bug，需要修复后使测试全部通过。

## 已知问题

- 模式解析完全没做只能精确匹配
- NFA构建完全没做状态转换都是硬编码的
- NFA匹配完全没做匹配要回溯很久
- DFA构建完全没做不能把NFA转换成DFA
- 捕获组完全没做匹配到了也不知道是哪个组

## 运行测试

```bash
python -m pytest tests/test_regex_engine.py -q
```

修复目标：所有 14 个测试全部通过。

## 技术要求

- 纯Python标准库，不引入第三方依赖
- 不修改测试文件
- 保持现有API接口不变
