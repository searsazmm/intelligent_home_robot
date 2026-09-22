# CLAUDE.md

居家陪伴机器人项目（Django 5.2 原型 + 三模块 TCP 架构，接口规范见 api_doc.md）。

## Agent skills

### Issue tracker

Issues 存放在 GitHub Issues（github.com/searsazmm/intelligent_home_robot），通过 gh CLI 读写。See `docs/agents/issue-tracker.md`.

### Triage labels

使用五个标准 triage 角色标签（needs-triage / needs-info / ready-for-agent / ready-for-human / wontfix）。See `docs/agents/triage-labels.md`.

### Domain docs

单上下文布局：根目录 CONTEXT.md + docs/adr/。See `docs/agents/domain.md`.
