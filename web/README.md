# Monitoring Gerbang - CSS

## CSS import order in index.html

```html
<link rel="stylesheet" href="assets/css/variables.css">
<link rel="stylesheet" href="assets/css/global.css">
<link rel="stylesheet" href="assets/css/sidebar.css">
<link rel="stylesheet" href="assets/css/header.css">
<link rel="stylesheet" href="assets/css/dashboard.css">
<link rel="stylesheet" href="assets/css/monitoring.css">
<link rel="stylesheet" href="assets/css/responsive.css">
```

## Required layout hooks

- Sidebar: `<aside class="sidebar" id="sidebar">`
- Header: `<header> ... </header>`
- Main: `<main id="dashboard">`
- Main content column: `<div class="dashboard-main">`
- Right panel: `<aside class="dashboard-aside">`
- CCTV wrapper: `<div class="well" id="well">`
- Activity section: `<section class="logbox" id="activity-logs">`

The sidebar width remains 240px at all breakpoints. On narrow screens,
the content area can become horizontally constrained; the sidebar is not
converted to an overlay or collapsed mode.
