# JAX blog source

This is the repository that will host the JAX team research blog.

## Contributing

Instructions for JAX team members to contribute new posts;
**please note that we will generally not accept community article contributions.**

Please contact @jakevdp or @skye if you have article ideas.

To create a new article, add a markdown file to `posts/` with appropriate metadata.
You may refer to  `posts/example-post.md` for an example, including how to format
code and mathematical text.

### Lint checks

To locally run the file linting checks done in the github CI, you can run
```bash
$ uv run pre-commit run --all-files
```

### Local preview

To build and preview your changes locally, you can run the following:
```bash
$ uv run mkdocs serve
```
If it builds successfully, the output should look something like this:
```
INFO    -  Building documentation...
INFO    -  Cleaning site directory
INFO    -  Documentation built in 0.24 seconds
INFO    -  [07:45:13] Serving on http://127.0.0.1:8000/
```
This output includes the localhost URL at which the site can be previewed.
Unlike the deployed site, this local preview will include posts marked with
`draft: true`.
