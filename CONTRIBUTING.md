# Flujo de trabajo

## Gitflow en resumen
- `main`: versión estable. Solo recibe merges desde `release/*` o `hotfix/*`.
- `develop`: integración de todo lo que ya está listo.
- `feature/<nombre>`: una tarea. Se crea desde `develop` y vuelve a `develop` por Pull Request.
- `release/<version>`: preparación de una versión. Sale de `develop`, termina en `main` y `develop`.
- `hotfix/<nombre>`: arreglo urgente sobre `main`.

## Ciclo normal (lo que haces tú)
```powershell
git checkout develop
git checkout -b feature/loader-premier-league
# ...Claude trabaja y deja commits en esa rama...
git log --oneline develop..feature/loader-premier-league   # revisa los commits
git push -u origin feature/loader-premier-league           # lo subes tú
# abres el Pull Request hacia develop en GitHub, lo revisas y haces merge
```

Claude **no** hace `push`, **no** abre PRs y **no** hace merge. Eso queda bloqueado en `.claude/settings.json` y también escrito en `CLAUDE.md`.

## Conventional Commits
`tipo(alcance): descripción`
- `feat`: funcionalidad nueva · `fix`: corrección · `docs`: documentación · `refactor`: reorganizar sin cambiar comportamiento
- `test`: pruebas · `chore`: tareas de mantenimiento · `perf`: rendimiento · `build`/`ci`: construcción e integración continua

Ejemplos:
- `feat(data): add premier league csv loader`
- `fix(features): use only matches before kickoff`
- `docs(readme): add reproduction steps`
