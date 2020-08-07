import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { CarouselvisiteurComponent } from './carousel.component';

describe('CarouselComponent', () => {
  let component: CarouselvisiteurComponent;
  let fixture: ComponentFixture<CarouselvisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ CarouselvisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(CarouselvisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
