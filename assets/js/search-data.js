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
        },{id: "post-attention-zoo-summary-of-ssms-and-transformers",
      
        title: "Attention Zoo: Summary of SSMs and Transformers",
      
      description: "An interactive guide to modern sequence models, explore architectures and recurrences across the linear-softmax landscape.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2026/attention-zoo/";
        
      },
    },{id: "post-raven-part-2",
      
        title: "Raven (Part-2)",
      
      description: "Architecture and Results",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2026/raven-part2/";
        
      },
    },{id: "post-raven-part-1",
      
        title: "Raven (Part-1)",
      
      description: "Memory as a set of Slots",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2026/raven-part1/";
        
      },
    },{id: "post-on-the-legacy-of-linear-transformers-in-positional-embeddings",
      
        title: "On the Legacy of Linear Transformers in Positional Embeddings 📍",
      
      description: "Duality of Forget Gates and Position Embeddings in Sequence Modeling",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2026/pe/";
        
      },
    },{id: "post-lion-part-iv-results",
      
        title: "LION 🦁 Part IV - Results",
      
      description: "Comprehensive results on Vision, MLM and more LION variants",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/lion-part4-results/";
        
      },
    },{id: "post-lion-part-iii-chunkwise-parallel-form-of-lion",
      
        title: "LION 🦁 Part III - Chunkwise Parallel Form of LION",
      
      description: "Explaining LION-Chunk for Balancing Memory-Speed Tradeoffs During Inference",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/lion-part3-chunk/";
        
      },
    },{id: "post-lion-part-ii-bi-directional-rnn",
      
        title: "LION 🦁 Part II - Bi-directional RNN",
      
      description: "Deriving equivalent bi-directional RNN for Linear Attention",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/lion-part2-theory/";
        
      },
    },{id: "post-lion-part-i-full-linear-attention",
      
        title: "LION 🦁 Part I - Full Linear Attention",
      
      description: "Explaining the Full Linear Attention paradigm for bi-directional sequence modeling",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/lion-part1-model/";
        
      },
    },{id: "news-rest-was-accepted-at-icml",
          title: 'REST was accepted at ICML! 🤩',
          description: "",
          section: "News",},{id: "news-best-poster-award-of-neuro-x",
          title: 'Best Poster award of Neuro-X!',
          description: "",
          section: "News",},{id: "news-lion-is-out-check-out-our-blog-post-for-an-in-depth-look-and-explore-our-new-bi-directional-linear-attention-framework-on-arxiv",
          title: '🦁 LION is out! Check out our blog post for an in-depth look,...',
          description: "",
          section: "News",},{id: "news-thrilled-to-receive-an-internship-offer-from",
          title: '🎉 Thrilled to receive an internship offer from !',
          description: "",
          section: "News",},{id: "news-lion-is-accepted-to-neurips-2025",
          title: '🦁 LION is accepted to NeuRIPS 2025!🍾',
          description: "",
          section: "News",},{id: "news-our-paper-selective-rotary-position-embedding-got-accepted-at-iclr-2026-see-you-in-rio",
          title: '🇧🇷 Our paper Selective Rotary Position Embedding got accepted at ICLR 2026! See...',
          description: "",
          section: "News",},{id: "news-selected-for-the-phd-summit-at-coitadel-honored-to-be-among-the-participants",
          title: '🏰 Selected for the PhD Summit at Coitadel! Honored to be among the...',
          description: "",
          section: "News",},{id: "news-️-new-blog-post-on-the-legacy-of-linear-transformers-as-positional-embeddings-exploring-the-duality-of-forget-gates-and-position-encodings",
          title: '✍️ New blog post: On the Legacy of Linear Transformers as Positional Embeddings...',
          description: "",
          section: "News",},{id: "news-m-excited-to-join-mistral-ai-as-a-research-intern",
          title: 'M Excited to join Mistral AI as a research intern!',
          description: "",
          section: "News",},{id: "news-raven-is-out-our-new-sequence-model-with-sparse-memory-routing-is-live-check-out-the-paper-and-the-blog-post-for-an-in-depth-walkthrough",
          title: '🐦‍⬛ Raven is out! Our new sequence model with sparse memory routing is...',
          description: "",
          section: "News",},{id: "news-honored-to-be-invited-to-yc-startup-school-2026",
          title: '🎉 Honored to be invited to  YC Startup School 2026!',
          description: "",
          section: "News",},{id: "news-we-gave-a-talk-on-our-paper-raven-at-download-the-slides",
          title: '🐦‍⬛ We gave a talk on our paper Raven at ! Download the...',
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
