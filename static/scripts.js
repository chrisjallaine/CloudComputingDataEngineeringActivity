// static/scripts.js
document.addEventListener('DOMContentLoaded', () => {
    // Add subtle animation to metrics
    const metrics = document.querySelectorAll('[data-testid="metric-container"]');
    metrics.forEach(metric => {
        metric.style.transform = 'translateY(20px)';
        metric.style.opacity = '0';
        setTimeout(() => {
            metric.style.transform = 'translateY(0)';
            metric.style.opacity = '1';
        }, 300);
    });

    // Chart hover effects
    document.querySelectorAll('.plot-container').forEach(chart => {
        chart.addEventListener('mouseenter', () => {
            chart.style.transform = 'scale(1.005)';
        });
        chart.addEventListener('mouseleave', () => {
            chart.style.transform = 'scale(1)';
        });
    });

    // Dynamic title update
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        const title = document.querySelector('h1');
        if (currentScroll > lastScroll) {
            title.style.opacity = '0.8';
        } else {
            title.style.opacity = '1';
        }
        lastScroll = currentScroll;
    });
});

// Performance monitoring
window.addEventListener('load', () => {
    const timing = performance.timing;
    const loadTime = timing.loadEventEnd - timing.navigationStart;
    console.log(`Dashboard loaded in ${loadTime}ms`);
});