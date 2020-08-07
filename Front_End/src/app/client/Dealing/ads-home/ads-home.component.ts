import { Component, OnInit } from '@angular/core';
import { NgbCarouselConfig } from '@ng-bootstrap/ng-bootstrap';

@Component({
  selector: 'app-ads-home',
  templateUrl: './ads-home.component.html',
  styleUrls: ['./ads-home.component.css']
})
export class AdsHomeComponent implements OnInit {
  images = [
    "assets/images/Ads/a.png",
    "assets/images/Ads/b.png",
    "assets/images/Ads/c.png",
    "assets/images/Ads/d.png" 
  ];
  constructor(config: NgbCarouselConfig) {
    config.interval = 2000; } 
   
  ngOnInit() {
  }

}
