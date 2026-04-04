# Team Member Images Setup

## Overview
Team member images are fetched from Google Drive using direct download links. This approach avoids storage limitations on both GitHub and Hugging Face.

## Team Member Images (Google Drive Links)

| Team Member | Role | Google Drive Link |
|---|---|---|
| Saptarshi Ghosh | Team Lead & Research Coordinator | https://drive.google.com/file/d/1Rp8eKsLqvN3AJ4y_52epKEAbU8yKXhht/view?usp=sharing |
| Sumit Chaira | UI/Deployment & Visualization Developer | https://drive.google.com/file/d/1jZlfSUCH8p2CV3ckjt3qVNbbLs5tFw4F/view?usp=sharing |
| Uday Shankar Dey | GNN Model Developer | https://drive.google.com/file/d/1_bROW_VquM1C6d27uOBHd7-0VZI5ivl8/view?usp=sharing |
| Mangaldip Dhua | Data Engineer & Preprocessing Specialist | https://drive.google.com/file/d/1CM8c2F09Je8HZ_Cyk5BsyOxp6vg7ZhWA/view?usp=sharing |
| Arnab Subhra Ghosh | Model Evaluation & Optimization Engineer | https://drive.google.com/file/d/1l5YV0yiWK_KMAgd37xsU6-6_blFXv05D/view?usp=sharing |

## How It Works

The HTML template uses direct Google Drive image URLs:
```html
<img src="https://drive.google.com/uc?export=view&id={FILE_ID}" alt="Team Member Name">
```

This approach:
- ✅ Eliminates git storage concerns
- ✅ Works on both GitHub and Hugging Face
- ✅ Keeps images accessible from anywhere
- ✅ No local file storage needed

## Advantages

1. **No Git Storage Overhead** - Images aren't stored in git repositories
2. **Universal Access** - Works on GitHub, Hugging Face, and local development
3. **Easy Updates** - Replace Google Drive files without code changes (same file IDs)
4. **Responsive** - Images load dynamically from Google Drive

## Local Development

For local development, images will load directly from Google Drive URLs. Ensure you have internet access.

## If Images Don't Load

1. Verify the Google Drive file IDs are correct
2. Check the files are shared with "Anyone with link" permission
3. Ensure internet connectivity

## Adding New Team Members

To add a new team member:
1. Upload their image to Google Drive
2. Get the file ID from the share link: `https://drive.google.com/file/d/{FILE_ID}/view`
3. Add a new team card in `templates/about.html` with the Google Drive URL

## Notes

- The `.gitignore` file excludes `static/team/` directory
- Images are served directly from Google Drive
- This eliminates binary file storage from both repositories

