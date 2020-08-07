import { Component, OnInit } from '@angular/core';
import { NgbCarouselConfig } from '@ng-bootstrap/ng-bootstrap';

@Component({
  selector: 'app-carousel-visiteur',
  templateUrl: './carousel.component.html',
  styleUrls: ['./carousel.component.css']
})
export class CarouselvisiteurComponent implements OnInit {
 /* The images should be 900x500*/ /* you can change it with css with % etc*/
 images = [
  "assets/images/Header/shoes.jpg",
  "assets/images/Header/phones.jpg",
  "assets/images/Header/watche.jpg"
];
  constructor(config: NgbCarouselConfig) { 
    config.interval = 3000;

  }

  ngOnInit() {
  }

}
