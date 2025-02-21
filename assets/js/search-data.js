// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "My Publications.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "Here is my CV! You can also use the pdf download button to have the full pdf version :)",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-lion-part-3-chunkwise-parallel-from-of-lion",
      
        title: "LION 🦁 Part 3 - Chunkwise Parallel from of LION",
      
      description: "Explaining LION-Chunk for Balancing Memory-Speed Tradeoffs During Inference",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/mamba2-part3-algorithm/";
        
      },
    },{id: "post-lion-part-2-bi-directional-rnn",
      
        title: "LION 🦁 Part 2 - Bi-directional RNN",
      
      description: "Deriving equivalent bidirectional RNN for linear Attention",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/mamba2-part2-theory/";
        
      },
    },{id: "post-lion-part-1-full-linear-attention",
      
        title: "LION 🦁 Part 1 - Full Linear Attention",
      
      description: "Explaining the full linear attention paradigm for bi-directional sequence modeling",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/mamba2-part1-model/";
        
      },
    },{id: "post-lion-part-4-results",
      
        title: "LION 🦁 Part 4 - Results",
      
      description: "Comprehensive results of LION on Vision, MLM and LION variants",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2024/mamba2-part4-results/";
        
      },
    },{id: "news-rest-was-accepted-at-icml",
          title: 'REST was accepted at ICML! 🤩',
          description: "",
          section: "News",},{id: "news-best-poster-award-of-neuro-x",
          title: 'Best Poster award of Neuro-X!',
          description: "",
          section: "News",},{id: "news-lion-will-be-out-soon-stay-tuned-for-our-new-bi-directional-linear-attention-framework",
          title: '🦁 LION will be out soon… Stay tuned for our new bi-directional linear...',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%61%72%73%68%69%61.%61%66%7A%61%6C@%65%70%66%6C.%63%68", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/arshiaafzal", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=OJ45nEQAAAAJ", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/rshia_afz", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
